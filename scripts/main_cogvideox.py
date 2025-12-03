import os, argparse, sys
sys.path.append("../Rectified-SpaAttn")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
from datetime import datetime
import time

from rectified_spaattn.rectified_cogvideo_attn import RectifiedCogVideoXVideoSpaAttnProcessor2_0
from utils.jenga_gilbert import gilbert_mapping, gilbert_block_neighbor_mapping
from utils.seed import set_seed
import utils.variable as va

from typing import Any, Dict, Optional, Tuple,  Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers, export_to_video, load_image
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline

# TeaCache
coefficients_dict = {
    "CogVideoX-2b":[-3.10658903e+01,  2.54732368e+01, -5.92380459e+00,  1.75769064e+00, -3.61568434e-03],
    "CogVideoX-5b":[-1.53880483e+03,  8.43202495e+02, -1.34363087e+02,  7.97131516e+00, -5.23162339e-02],
    "CogVideoX-5b-I2V":[-1.53880483e+03,  8.43202495e+02, -1.34363087e+02,  7.97131516e+00, -5.23162339e-02],
    "CogVideoX1.5-5B":[ 2.50210439e+02, -1.65061612e+02,  3.57804877e+01, -7.81551492e-01, 3.58559703e-02],
    "CogVideoX1.5-5B-I2V":[ 1.22842302e+02, -1.04088754e+02,  2.62981677e+01, -3.06009921e-01, 3.71213220e-02],
}


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
        curve_sels.append([torch.tensor(LINEAR_TO_HILBERT, dtype=torch.long, device='cuda'), torch.tensor(HILBERT_ORDER, dtype=torch.long, device='cuda'), block_neighbor_list])

    return curve_sels


def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
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
                print(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # Jenga
        hidden_states = hidden_states[:, self.hilbert_order]
        image_rotary_emb = (image_rotary_emb[0][self.hilbert_order], image_rotary_emb[1][self.hilbert_order])

        if self.cnt == 0:
            import utils.variable as va
            torch.cuda.synchronize()
            va.time_start = time.time()

        if self.enable_teacache:
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((emb-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = emb
        
        self.cnt += 1          
        # set current_step
        for name, processor in self.attn_processors.items():
            if isinstance(processor, RectifiedCogVideoXVideoSpaAttnProcessor2_0):
                processor.current_step = self.cnt-1

        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
                encoder_hidden_states += self.previous_residual_encoder
            else:
                ori_hidden_states = hidden_states.clone()
                ori_encoder_hidden_states = encoder_hidden_states.clone()
                # 4. Transformer blocks
                for i, block in enumerate(self.transformer_blocks):
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )

                self.previous_residual = hidden_states - ori_hidden_states
                self.previous_residual_encoder = encoder_hidden_states - ori_encoder_hidden_states
        else:
            # 4. Transformer blocks
            for i, block in enumerate(self.transformer_blocks):
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

        # Jenga
        hidden_states = hidden_states[:, self.linear_to_hilbert]

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B and CogvideoX1.5-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 5. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

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


def main(args):
    seed = args.seed
    set_seed(seed)
    rel_l1_thresh = args.rel_l1_thresh
    generate_type = args.generate_type
    negative_prompt = args.negative_prompt
    num_inference_steps = args.num_inference_steps
    height = args.height
    width = args.width
    num_frames = args.num_frames
    guidance_scale = args.guidance_scale
    fps = args.fps

    if generate_type == "t2v":
        ckpts_path = "THUDM/CogVideoX1.5-5B"
        prompt = args.prompt_t2v
        pipe = CogVideoXPipeline.from_pretrained(ckpts_path, torch_dtype=torch.bfloat16)
    else:
        ckpts_path = "THUDM/CogVideoX1.5-5B-I2V"
        prompt = args.prompt_i2v
        image_path = args.image_path
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(ckpts_path, torch_dtype=torch.bfloat16)
        image = load_image(image=image_path)

    mode = ckpts_path.split("/")[-1]
    # Jenga
    latent_time_, latent_height_, latent_width_ = (num_frames-1) // 8 + 1, height // 16, width // 16
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
    pipe.transformer.__class__.rel_l1_thresh = rel_l1_thresh
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None
    pipe.transformer.__class__.previous_residual_encoder = None
    pipe.transformer.__class__.num_steps = num_inference_steps
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.coefficients = coefficients_dict[mode]

    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Jenga
    pipe.transformer.__class__.forward = teacache_forward
    pipe.transformer.__class__.linear_to_hilbert = hilbert_params[0][0]
    pipe.transformer.__class__.hilbert_order = hilbert_params[0][1]
    pipe.transformer.__class__.hilbert_params = hilbert_params

    attn_processors = {}
    processors_id = 0
    for k, v in pipe.transformer.attn_processors.items():
        attn_processors[k] = RectifiedCogVideoXVideoSpaAttnProcessor2_0(args.mode, select_block_num, hilbert_params[0][2], p_remain_rates, processors_id)
        processors_id += 1  
        
    pipe.transformer.set_attn_processor(attn_processors)

    if generate_type == "t2v":
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cuda").manual_seed(seed)
        ).frames[0]
    else:
        video = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]

    time_flag = datetime.fromtimestamp(time.time()).strftime("%m-%d-%H:%M:%S")
    cur_save_path = f"./{time_flag}_cogvideox1.5_{generate_type}_{va.time_end - va.time_start:.4f}.mp4"
    export_to_video(video, cur_save_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CogVideoX1.5-5B with given parameters")
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--negative_prompt', type=str, default=None, help='Description of unwanted situations in model generated videos')
    parser.add_argument("--width", type=int, default=1280, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=768, help="Number of steps for the inference process")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of steps for the inference process")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--fps", type=int, default=16, help="Frame rate of video")
    parser.add_argument('--prompt_t2v', type=str, default="A clear, turquoise river flows through a rocky canyon, cascading over a small waterfall and forming a pool of water at the bottom.The river is the main focus of the scene, with its clear water reflecting the surrounding trees and rocks. The canyon walls are steep and rocky, with some vegetation growing on them. The trees are mostly pine trees, with their green needles contrasting with the brown and gray rocks. The overall tone of the scene is one of peace and tranquility.", help='Description of the text-2-video for the model to generate')
    parser.add_argument('--prompt_i2v', type=str, default="a boat sits on the shore of a lake with mt fuji in the background.", help='Description of the image-2-video for the model to generate')
    parser.add_argument("--image_path",type=str,default='./assets/a boat sits on the shore of a lake with mt fuji in the background.jpg',help="The path of the image to be used as the background of the video")
    parser.add_argument("--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v')")

    parser.add_argument("--sa_drop_rate", type=float, default=0.75, help="Drop rate for sparse attention.")
    parser.add_argument("--p_remain_rates", type=float, default=0.3, help="Probability for keeping tokens.")
    parser.add_argument("--enable_teacache", action="store_true", help="Enable teacher-cache acceleration.") # , default=True
    parser.add_argument('--rel_l1_thresh', type=float, default=0.2, help='Higher speedup will cause to worse quality -- 0.1 for 1.3x speedup -- 0.2 for 1.8x speedup -- 0.3 for 2.1x speedup')
    parser.add_argument("--mode", type=str, default="sparse", choices=["sparse", "flash", "torch", "vanilla"], help="Inference mode.")

    args = parser.parse_args()

    main(args)