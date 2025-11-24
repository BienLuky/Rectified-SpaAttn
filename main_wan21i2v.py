import os, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import load_image
from transformers import CLIPVisionModel

from rectified_spaattn.rectified_wan21_attn import RectifiedWanI2VSpaAttnProcessor2_0
from rectified_spaattn.attn_processor import get_attn_processors, set_attn_processor
from utils.jenga_gilbert import gilbert_mapping, gilbert_block_neighbor_mapping
from utils.seed import set_seed
from utils.save_video import save_videos_grid
import utils.variable as va

import math
import numpy as np
from datetime import datetime
import time
from typing import Any, List, Tuple, Optional, Union, Dict
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def build_multi_curve(latent_time, latent_height, latent_width, axis_order_list):
    curve_sels = []
    latent_time_ = int(latent_time)
    latent_height_ = int(latent_height)
    latent_width_ = int(latent_width)
    if (latent_height_ * latent_width_) % 4 != 0:
        raise ValueError(f"latent_height_ * latent_width_ must be divisible by 4, but got {latent_height_ * latent_width_}")
    
    for axis_order in axis_order_list:
        LINEAR_TO_HILBERT, HILBERT_ORDER = gilbert_mapping(latent_time_, latent_height_, latent_width_, axis_order=axis_order)
        block_neighbor_list = gilbert_block_neighbor_mapping(latent_time_, latent_height_, latent_width_, axis_order=axis_order)

        # # linear settings.
        # LINEAR_TO_HILBERT = torch.arange(latent_time_ * latent_height_ * latent_width_, dtype=torch.long) # linear
        # HILBERT_ORDER = torch.arange(latent_time_ * latent_height_ * latent_width_, dtype=torch.long) # linear
        # block_neighbor_list = torch.zeros((math.ceil(latent_time_ * latent_height_ * latent_width_ / 128), math.ceil(latent_time_ * latent_height_ * latent_width_ / 128)), dtype=torch.bool)
        
        curve_sels.append([torch.tensor(LINEAR_TO_HILBERT, dtype=torch.long, device='cuda'), torch.tensor(HILBERT_ORDER, dtype=torch.long, device='cuda'), block_neighbor_list])

    return curve_sels


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    # 1. RoPE
    rotary_emb = self.rope(hidden_states)

    # 2. Conditional embeddings
    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    hidden_states = hidden_states[:, self.hilbert_order]
    rotary_emb = rotary_emb[:, :, self.hilbert_order]

    # 3. Attention mask preparation ----- is None

    if self.cnt == 0:
        import utils.variable as va
        torch.cuda.synchronize()
        va.time_start = time.time()
        
    if self.enable_teacache:
        modulated_inp = timestep_proj if self.use_ref_steps else temb
        # teacache
        if self.cnt%2==0: # even -> conditon
            self.is_even = True
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc_even = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp-self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    should_calc_even = False
                else:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()

        else: # odd -> unconditon
            self.is_even = False
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc_odd = True
                self.accumulated_rel_l1_distance_odd = 0
            else: 
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp-self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc_odd = False
                else:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()
    self.cnt += 1 

    if self.enable_teacache: 
        if self.is_even:
            if not should_calc_even:
                hidden_states += self.previous_residual_even
            else:
                ori_hidden_states = hidden_states.clone()
                # 4. Transformer blocks
                for block in self.blocks:
                    hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                self.previous_residual_even = hidden_states - ori_hidden_states
        else:
            if not should_calc_odd:
                hidden_states += self.previous_residual_odd
            else:
                ori_hidden_states = hidden_states.clone()
                # 4. Transformer blocks
                for block in self.blocks:
                    hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                self.previous_residual_odd = hidden_states - ori_hidden_states
    
    else:
        # 4. Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

    # Jenga
    hidden_states = hidden_states[:, self.linear_to_hilbert]

    # 5. Output norm, projection & unpatchify
    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    # Move the shift and scale tensors to the same device as hidden_states.
    # When using multi-GPU inference via accelerate these will be on the
    # first device rather than the last device, which hidden_states ends up
    # on.
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if self.cnt == self.num_steps:
        import utils.variable as va
        torch.cuda.synchronize()
        va.time_end = time.time()

    if self.cnt == self.num_steps:
        self.cnt = 0   

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


# use_ret_steps = False
# Wan2.1 i2v 480p	Slow (0.13)	Fast (0.26)
# Wan2.1 i2v 720p	Slow (0.20)	Fast (0.30)
# use_ret_steps = True
# Wan2.1 i2v 480p	Slow (0.20)	Fast (0.30)
# Wan2.1 i2v 720p	Slow (0.20)	Fast (0.30)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=720, help="video height.")
    parser.add_argument("--width", type=int, default=1280, help="video width.")
    parser.add_argument("--frame", type=int, default=81, help="video frames.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of diffusion sampling steps.")
    parser.add_argument("--sa_drop_rate", type=float, default=0.75, help="Drop rate for sparse attention.")
    parser.add_argument("--p_remain_rates", type=float, default=0.3, help="Probability for keeping tokens.")
    parser.add_argument("--enable_teacache", action="store_true", default=False, help="Enable teacher-cache acceleration.")
    parser.add_argument("--teacache_thresh", type=float, default=0.3, help="TeaCache thresh sparams.")
    parser.add_argument("--mode", type=str, default="sparse", choices=["sparse", "flash", "torch", "vanilla"], help="Inference mode.")
    parser.add_argument("--use_ret_steps",action="store_true",default=True,
        help="Using Retention Steps will result in faster generation speed and better generation quality.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(42)

    # Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
    model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    height, width, frame = args.height, args.width, args.frame
    image = load_image("./assets/a boat sits on the shore of a lake with mt fuji in the background.jpg")
    prompt = (
        "a boat sits on the shore of a lake with mt fuji in the background."
    )
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    max_area = height * width
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    image_height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    image_width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((image_width, image_height))

    # Jenga
    latent_time_, latent_height_, latent_width_ = (frame + 3) // 4, height // 16, width // 16
    axis_order_list = [("w","h","t")]
    hilbert_params = build_multi_curve(latent_time_, latent_height_, latent_width_, axis_order_list)

    # Prepare sparse params
    img_seq_len = latent_time_ * latent_height_ * latent_width_
    sa_drop_rate = args.sa_drop_rate
    block_size = 128
    img_block_num = (img_seq_len + block_size - 1) // block_size
    select_block_num = int((1 - sa_drop_rate) * img_block_num)
    p_remain_rates = args.p_remain_rates
    first_frame_blocks = math.ceil(img_block_num // latent_time_) # following prior works-keep first frame

    # TeaCache
    pipe.transformer.__class__.enable_teacache = args.enable_teacache
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = args.num_steps * 2
    pipe.transformer.__class__.teacache_thresh = args.teacache_thresh 
    pipe.transformer.__class__.accumulated_rel_l1_distance_even = 0
    pipe.transformer.__class__.accumulated_rel_l1_distance_odd = 0
    pipe.transformer.__class__.previous_e0_even = None
    pipe.transformer.__class__.previous_e0_odd = None
    pipe.transformer.__class__.previous_residual_even = None
    pipe.transformer.__class__.previous_residual_odd = None
    pipe.transformer.__class__.use_ref_steps = args.use_ret_steps
    if args.use_ret_steps:
        if '1.3B' in model_id:
            pipe.transformer.__class__.coefficients = [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02]
        if '14B' in model_id:
            pipe.transformer.__class__.coefficients = [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]
        pipe.transformer.__class__.ret_steps = 5*2
        pipe.transformer.__class__.cutoff_steps = args.num_steps*2
    else:
        if '1.3B' in model_id:
            pipe.transformer.__class__.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
        if '14B' in model_id:
            pipe.transformer.__class__.coefficients = [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
        pipe.transformer.__class__.ret_steps = 1*2
        pipe.transformer.__class__.cutoff_steps = args.num_steps*2 - 2

    # Jenga
    pipe.transformer.__class__.forward = teacache_forward
    pipe.transformer.__class__.linear_to_hilbert = hilbert_params[0][0]
    pipe.transformer.__class__.hilbert_order = hilbert_params[0][1]

    attn_processors = {}
    processors_id = 0
    for k,v in get_attn_processors(pipe.transformer).items():
        if "attn1" in k:
            attn_processors[k] = RectifiedWanI2VSpaAttnProcessor2_0(args.mode, select_block_num, hilbert_params[0][2], p_remain_rates, processors_id, first_frame_blocks)
        elif "attn2" in k:
            attn_processors[k] = RectifiedWanI2VSpaAttnProcessor2_0("flash", select_block_num, hilbert_params[0][2], p_remain_rates, processors_id)
            processors_id += 1   
    set_attn_processor(pipe.transformer, attn_processors)

    outputs = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=frame,
        guidance_scale=5.0,
        num_inference_steps=args.num_steps
    ).frames

    for i, sample in enumerate(outputs):
        time_flag = datetime.fromtimestamp(time.time()).strftime("%m-%d-%H:%M:%S")
        cur_save_path = f"./{time_flag}_wan21i2v_{va.time_end - va.time_start:.0f}s.mp4"
        save_videos_grid(sample, cur_save_path, fps=16)    
        print(f'Sample save to: {cur_save_path}')
