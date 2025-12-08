import os, argparse, sys
sys.path.append("../Rectified-SpaAttn")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from rectified_spaattn.rectified_wan22_attn import RectifiedWanTI2VSpaAttnProcessor2_0, RectifiedWanT2VSpaAttnProcessor2_0, RectifiedWanI2VSpaAttnProcessor2_0
from rectified_spaattn.attn_processor import get_attn_processors, set_attn_processor
from utils.jenga_gilbert import gilbert_mapping, gilbert_block_neighbor_mapping
from utils.seed import set_seed
import utils.variable as va

import math
import numpy as np
from datetime import datetime
import time
from typing import Any, List, Tuple, Optional, Union, Dict
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers, export_to_video, load_image
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

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Jenga
        hidden_states = hidden_states[:, self.hilbert_order]
        rotary_emb = (rotary_emb[0][:, self.hilbert_order], rotary_emb[1][:, self.hilbert_order])

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

        # Set current_step
        for name, processor in get_attn_processors(self.blocks).items():
            if isinstance(processor, (RectifiedWanTI2VSpaAttnProcessor2_0, RectifiedWanT2VSpaAttnProcessor2_0, RectifiedWanI2VSpaAttnProcessor2_0)):
                processor.current_step = self.cnt-1

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
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

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


# TeaCache params
# use_ret_steps = False
# Wan2.1 t2v 1.3B	Slow (0.05)	Fast (0.08)
# Wan2.1 t2v 14B	Slow (0.14)	Fast (0.20)
# use_ret_steps = True
# Wan2.1 t2v 1.3B	Slow (0.05)	Fast (0.10)
# Wan2.1 t2v 14B	Slow (0.10)	Fast (0.20)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=704, help="video height.")
    parser.add_argument("--width", type=int, default=1280, help="video width.")
    parser.add_argument("--frame", type=int, default=121, help="video frames.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of diffusion sampling steps.")
    parser.add_argument("--sa_drop_rate", type=float, default=0.75, help="Drop rate for sparse attention.")
    parser.add_argument("--p_remain_rates", type=float, default=0.3, help="Probability for keeping tokens.")
    parser.add_argument("--enable_teacache", action="store_true", help="Enable teacher-cache acceleration.")
    parser.add_argument("--teacache_thresh", type=float, default=0.1, help="TeaCache thresh sparams.")
    parser.add_argument("--mode", type=str, default="sparse", choices=["sparse", "flash", "torch", "vanilla"], help="Inference mode.")
    parser.add_argument("--use_ret_steps",action="store_true",default=True,
        help="Using Retention Steps will result in faster generation speed and better generation quality.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(42)

    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")


    height, width, frame = args.height, args.width, args.frame
    prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    # Jenga
    latent_time_, latent_height_, latent_width_ = (frame + 3) // 4, height // 32, width // 32
    axis_order_list = [("w","h","t")]
    hilbert_params = build_multi_curve(latent_time_, latent_height_, latent_width_, axis_order_list)

    # Prepare sparse params
    img_seq_len = latent_time_ * latent_height_ * latent_width_
    sa_drop_rate = args.sa_drop_rate
    block_size = 128
    img_block_num = (img_seq_len + block_size - 1) // block_size
    select_block_num = int((1 - sa_drop_rate) * img_block_num)
    p_remain_rates = args.p_remain_rates
    first_frame_blocks = math.ceil(img_block_num // latent_time_) * 1  # following prior works-keep first frame

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
        if '14B' in model_id or '5B' in model_id:
            pipe.transformer.__class__.coefficients = [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]
        pipe.transformer.__class__.ret_steps = 5*2
        pipe.transformer.__class__.cutoff_steps = args.num_steps*2
    else:
        if '1.3B' in model_id:
            pipe.transformer.__class__.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
        if '14B' in model_id or '5B' in model_id:
            pipe.transformer.__class__.coefficients = [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
        pipe.transformer.__class__.ret_steps = 1*2
        pipe.transformer.__class__.cutoff_steps = args.num_steps*2 - 2

    # # Jenga
    pipe.transformer.__class__.forward = teacache_forward
    pipe.transformer.__class__.linear_to_hilbert = hilbert_params[0][0]
    pipe.transformer.__class__.hilbert_order = hilbert_params[0][1]

    attn_processors = {}
    processors_id = 0
    for k,v in get_attn_processors(pipe.transformer).items():
        if "attn1" in k:
            attn_processors[k] = RectifiedWanTI2VSpaAttnProcessor2_0(args.mode, select_block_num, hilbert_params[0][2], p_remain_rates, processors_id, first_frame_blocks)
        elif "attn2" in k:
            attn_processors[k] = RectifiedWanTI2VSpaAttnProcessor2_0("flash", select_block_num, hilbert_params[0][2], p_remain_rates, processors_id) 
            processors_id += 1  
    set_attn_processor(pipe.transformer, attn_processors)

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=frame,
        guidance_scale=5.0,
        num_inference_steps=args.num_steps
    ).frames[0]

    time_flag = datetime.fromtimestamp(time.time()).strftime("%m-%d-%H:%M:%S")
    cur_save_path = f"./{time_flag}_wan22ti2v_{va.time_end - va.time_start:.0f}s.mp4"
    export_to_video(output, cur_save_path, fps=24)
    print(f'Sample save to: {cur_save_path}')
