import os, argparse, sys
sys.path.append("../Rectified-SpaAttn")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

from rectified_spaattn.rectified_hunyuan_attn import RectifiedHunyuanVideoSpaAttnProcessor2_0
from utils.jenga_gilbert import gilbert_mapping, gilbert_block_neighbor_mapping
from utils.seed import set_seed
from utils.save_video import save_videos_grid
import utils.variable as va

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
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
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
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width

    # 1. RoPE
    image_rotary_emb = self.rope(hidden_states)

    # 2. Conditional embeddings
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    # Jenga
    hidden_states = hidden_states[:, self.hilbert_order]
    image_rotary_emb = (image_rotary_emb[0][self.hilbert_order], image_rotary_emb[1][self.hilbert_order])
    
    # 3. Attention mask preparation
    latent_sequence_length = hidden_states.shape[1]
    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.ones(
        batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
    )  # [B, N]
    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
    indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)  # [1, N]
    mask_indices = indices >= effective_sequence_length.unsqueeze(1)  # [B, N]
    attention_mask = attention_mask.masked_fill(mask_indices, False)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]

    if self.cnt == 0:
        import utils.variable as va
        torch.cuda.synchronize()
        va.time_start = time.time()

    if self.enable_teacache:
        inp = hidden_states.clone()
        temb_ = temb.clone()
        modulated_inp, _, _, _, _ = self.transformer_blocks[0].norm1(inp, emb=temb_)
        if self.cnt == 0 or self.cnt == self.num_steps-1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else: 
            coefficients = [7.33226126e+02, -4.01131952e+02,  6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp  
    self.cnt += 1  

    if self.enable_teacache:
        if not should_calc:
            print(f"skip self.cnt: {self.cnt}")
            hidden_states += self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            # 4. Transformer blocks
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )
            self.previous_residual = hidden_states - ori_hidden_states
    else:        
        # 4. Transformer blocks
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

    # Jenga
    hidden_states = hidden_states[:, self.linear_to_hilbert]

    # 5. Output projection
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

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
        return (hidden_states,)

    return Transformer2DModelOutput(sample=hidden_states)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=720, help="video height.")
    parser.add_argument("--width", type=int, default=1280, help="video width.")
    parser.add_argument("--frame", type=int, default=128, help="video frames.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of diffusion sampling steps.")
    parser.add_argument("--sa_drop_rate", type=float, default=0.8, help="Drop rate for sparse attention.")
    parser.add_argument("--p_remain_rates", type=float, default=0.3, help="Probability for keeping tokens.")
    parser.add_argument("--enable_teacache", action="store_true", help="Enable teacher-cache acceleration.")
    parser.add_argument("--rel_l1_thresh", type=float, default=0.15, help="TeaCache thresh sparams (0.1 for ~1.6x speedup, 0.15 for ~2.1x speedup).")
    parser.add_argument("--mode", type=str, default="sparse", choices=["sparse", "flash", "torch", "vanilla"], help="Inference mode.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(42)

    model_id = "hunyuanvideo-community/HunyuanVideo"
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
    device = torch.device('cuda')    
    pipe.vae.enable_tiling()
    # pipe.enable_model_cpu_offload()
    pipe.to(device)

    height, width, frame = args.height, args.width, args.frame
    prompt = ["several hot air balloons flying over a city."]

    # Jenga
    latent_time_, latent_height_, latent_width_ = frame // 4, height // 16, width // 16
    axis_order_list = [("w","h","t")]
    hilbert_params = build_multi_curve(latent_time_, latent_height_, latent_width_, axis_order_list)

    # Prepare sparse params
    img_seq_len = latent_time_ * latent_height_ * latent_width_
    sa_drop_rate = args.sa_drop_rate
    block_size = 128
    img_block_num = img_seq_len // block_size
    select_block_num = int((1 - sa_drop_rate) * img_block_num)
    p_remain_rates = args.p_remain_rates

    # TeaCache
    pipe.transformer.__class__.enable_teacache = args.enable_teacache
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = args.num_steps
    pipe.transformer.__class__.rel_l1_thresh = args.rel_l1_thresh 
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None

    # Jenga
    pipe.transformer.__class__.forward = teacache_forward
    pipe.transformer.__class__.linear_to_hilbert = hilbert_params[0][0]
    pipe.transformer.__class__.hilbert_order = hilbert_params[0][1]
        
    attn_processors = {}
    processors_id = 0
    for k,v in transformer.attn_processors.items():
        if "token_refiner" in k:
            attn_processors[k] = v
        else:
            attn_processors[k] = RectifiedHunyuanVideoSpaAttnProcessor2_0(args.mode, select_block_num, hilbert_params[0][2], p_remain_rates, processors_id)
            processors_id += 1   
    transformer.set_attn_processor(attn_processors)

    outputs = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=frame,
        num_inference_steps=args.num_steps
    ).frames

    for i, sample in enumerate(outputs):
        time_flag = datetime.fromtimestamp(time.time()).strftime("%m-%d-%H:%M:%S")
        cur_save_path = f"./{time_flag}_hunyuan_{va.time_end - va.time_start:.0f}s.mp4"
        save_videos_grid(sample, cur_save_path, fps=24)    
        print(f'Sample save to: {cur_save_path}')

