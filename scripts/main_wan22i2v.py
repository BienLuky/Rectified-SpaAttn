import os, argparse, sys
sys.path.append("../Rectified-SpaAttn")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from diffusers import WanImageToVideoPipeline

from rectified_spaattn.rectified_wan22_attn import RectifiedWanI2VSpaAttnProcessor2_0
from rectified_spaattn.attn_processor import get_attn_processors, set_attn_processor
from utils.seed import set_seed
import utils.variable as va
from scripts.main_wan22ti2v import build_multi_curve, teacache_forward

import types
import math
import numpy as np
from datetime import datetime
import time
from diffusers.utils import logging, export_to_video, load_image
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TeaCache params
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
    parser.add_argument("--num_steps", type=int, default=40, help="Number of diffusion sampling steps.")
    parser.add_argument("--sa_drop_rate", type=float, default=0.85, help="Drop rate for sparse attention.")
    parser.add_argument("--p_remain_rates", type=float, default=0.3, help="Probability for keeping tokens.")
    parser.add_argument("--enable_teacache", action="store_true", help="Enable teacher-cache acceleration.")
    parser.add_argument("--teacache_thresh", type=float, default=0.3, help="TeaCache thresh sparams.")
    parser.add_argument("--mode", type=str, default="flash", choices=["sparse", "flash", "torch", "vanilla"], help="Inference mode.")
    parser.add_argument("--use_ret_steps",action="store_true",default=True,
        help="Using Retention Steps will result in faster generation speed and better generation quality.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(42)

    model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    # 4. Prepare timesteps
    pipe.scheduler.set_timesteps(args.num_steps, device="cuda")
    timesteps = pipe.scheduler.timesteps
    boundary_timestep = pipe.config.boundary_ratio * pipe.scheduler.config.num_train_timesteps
    idx = torch.where(timesteps >= boundary_timestep)[0]
    transformer_steps = idx[-1].item() + 1
    
    height, width, frame = args.height, args.width, args.frame
    image = load_image("./assets/a boat sits on the shore of a lake with mt fuji in the background.jpg")
    prompt = (
        "a boat sits on the shore of a lake with mt fuji in the background."
    )
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

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
    first_frame_blocks = math.ceil(img_block_num // latent_time_) * 1  # following prior works-keep first frame

    # TeaCache
    # transformer
    pipe.transformer.enable_teacache = args.enable_teacache
    pipe.transformer.cnt = 0
    pipe.transformer.num_steps = transformer_steps * 2
    pipe.transformer.teacache_thresh = args.teacache_thresh 
    pipe.transformer.accumulated_rel_l1_distance_even = 0
    pipe.transformer.accumulated_rel_l1_distance_odd = 0
    pipe.transformer.previous_e0_even = None
    pipe.transformer.previous_e0_odd = None
    pipe.transformer.previous_residual_even = None
    pipe.transformer.previous_residual_odd = None
    pipe.transformer.use_ref_steps = args.use_ret_steps
    # transformer_2
    pipe.transformer_2.enable_teacache = args.enable_teacache
    pipe.transformer_2.cnt = transformer_steps * 2
    pipe.transformer_2.num_steps = args.num_steps * 2
    pipe.transformer_2.teacache_thresh = args.teacache_thresh 
    pipe.transformer_2.accumulated_rel_l1_distance_even = 0
    pipe.transformer_2.accumulated_rel_l1_distance_odd = 0
    pipe.transformer_2.previous_e0_even = None
    pipe.transformer_2.previous_e0_odd = None
    pipe.transformer_2.previous_residual_even = None
    pipe.transformer_2.previous_residual_odd = None
    pipe.transformer_2.use_ref_steps = args.use_ret_steps
    if args.use_ret_steps:
        if '1.3B' in model_id:
            pipe.transformer.coefficients = [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02]
            pipe.transformer_2.coefficients = [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02]
        if '14B' in model_id or '5B' in model_id:
            pipe.transformer.coefficients = [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]
            pipe.transformer_2.coefficients = [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]
        pipe.transformer.ret_steps = 3*2
        pipe.transformer.cutoff_steps = transformer_steps * 2
        pipe.transformer_2.ret_steps = transformer_steps * 2 + 1*2
        pipe.transformer_2.cutoff_steps = args.num_steps*2
    else:
        if '1.3B' in model_id:
            pipe.transformer.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
            pipe.transformer.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
        if '14B' in model_id or '5B' in model_id:
            pipe.transformer_2.coefficients = [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
            pipe.transformer_2.coefficients = [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
        pipe.transformer.ret_steps = 1*2
        pipe.transformer_2.cutoff_steps = transformer_steps * 2 - 2
        pipe.transformer.ret_steps = transformer_steps * 2 + 1*2
        pipe.transformer_2.cutoff_steps = args.num_steps*2 - 2

    # # Jenga
    pipe.transformer.forward = types.MethodType(teacache_forward, pipe.transformer)
    pipe.transformer.linear_to_hilbert = hilbert_params[0][0]
    pipe.transformer.hilbert_order = hilbert_params[0][1]

    pipe.transformer_2.forward = types.MethodType(teacache_forward, pipe.transformer_2)
    pipe.transformer_2.linear_to_hilbert = hilbert_params[0][0]
    pipe.transformer_2.hilbert_order = hilbert_params[0][1]

    attn_processors = {}
    processors_id = 0
    for k,v in get_attn_processors(pipe.transformer).items():
        if "attn1" in k:
            attn_processors[k] = RectifiedWanI2VSpaAttnProcessor2_0(args.mode, select_block_num, hilbert_params[0][2], p_remain_rates, processors_id, first_frame_blocks, warm_steps=0)
        elif "attn2" in k:
            attn_processors[k] = RectifiedWanI2VSpaAttnProcessor2_0("flash", select_block_num, hilbert_params[0][2], p_remain_rates, processors_id) 
            processors_id += 1  
    set_attn_processor(pipe.transformer, attn_processors)

    attn_processors = {}
    for k,v in get_attn_processors(pipe.transformer_2).items():
        if "attn1" in k:
            attn_processors[k] = RectifiedWanI2VSpaAttnProcessor2_0(args.mode, select_block_num, hilbert_params[0][2], p_remain_rates, processors_id, first_frame_blocks, warm_steps=transformer_steps * 2+0)
        elif "attn2" in k:
            attn_processors[k] = RectifiedWanI2VSpaAttnProcessor2_0("flash", select_block_num, hilbert_params[0][2], p_remain_rates, processors_id) 
            processors_id += 1  
    set_attn_processor(pipe.transformer_2, attn_processors)

    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=frame,
        guidance_scale=3.5,
        num_inference_steps=args.num_steps
    ).frames[0]

    time_flag = datetime.fromtimestamp(time.time()).strftime("%m-%d-%H:%M:%S")
    cur_save_path = f"./{time_flag}_wan22i2v_{va.time_end - va.time_start:.0f}s.mp4"
    export_to_video(output, cur_save_path, fps=16)
    print(f'Sample save to: {cur_save_path}')
