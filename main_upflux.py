import os, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import torch
from diffusers import FluxPipeline, FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline

from rectified_spaattn.rectified_flux_attn import RectifiedFluxSpaAttnProcessor2_0
from utils.jenga_gilbert import gilbert_mapping, gilbert_block_neighbor_mapping
from utils.seed import set_seed
import utils.variable as va

import numpy as np
from datetime import datetime
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
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # Jenga
    hidden_states = hidden_states[:, self.hilbert_order]
    controlnet_block_samples = [
        hs[:, self.hilbert_order] for hs in controlnet_block_samples
    ]
    text_rotary_emb0 = image_rotary_emb[0][:encoder_hidden_states.shape[1], ...]
    text_rotary_emb1 = image_rotary_emb[1][:encoder_hidden_states.shape[1], ...]
    image_rotary_emb0 = image_rotary_emb[0][encoder_hidden_states.shape[1]:, ...][self.hilbert_order]
    image_rotary_emb1 = image_rotary_emb[1][encoder_hidden_states.shape[1]:, ...][self.hilbert_order]
    image_rotary_emb = (torch.cat((image_rotary_emb0, text_rotary_emb0), dim=0), 
                        torch.cat((image_rotary_emb1, text_rotary_emb1), dim=0))

    if self.cnt == 0:
        import utils.variable as va
        torch.cuda.synchronize()
        va.time_start = time.time()

    if self.enable_teacache:
        inp = hidden_states.clone()
        temb_ = temb.clone()
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
        if self.cnt == 0 or self.cnt == self.num_steps-1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else: 
            coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
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
            hidden_states += self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            for index_block, block in enumerate(self.transformer_blocks):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

                # controlnet residual
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    # For Xlabs ControlNet.
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            # hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            # Jenga
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

                # controlnet residual
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )

            # hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
            # Jenga
            hidden_states = hidden_states[:, :-encoder_hidden_states.shape[1], ...]
            self.previous_residual = hidden_states - ori_hidden_states
    else:
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        # hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        # Jenga
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        # hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        # Jenga
        hidden_states = hidden_states[:, :-encoder_hidden_states.shape[1], ...]

    # Jenga
    hidden_states = hidden_states[:, self.linear_to_hilbert]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

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
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=1024, help="image height.")
    parser.add_argument("--width", type=int, default=1024, help="image width.")
    parser.add_argument("--upscale", type=int, default=4, help="up ratios.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of diffusion sampling steps.")
    parser.add_argument("--sa_drop_rate", type=float, default=0.9, help="Drop rate for sparse attention.")
    parser.add_argument("--p_remain_rates", type=float, default=0.3, help="Probability for keeping tokens.")
    parser.add_argument("--enable_teacache", action="store_true", default=True, help="Enable teacher-cache acceleration.")
    parser.add_argument("--rel_l1_thresh", type=float, default=0.8, help="TeaCache thresh sparams (0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup).")
    parser.add_argument("--mode", type=str, default="sparse", choices=["sparse", "flash", "torch", "vanilla"], help="Inference mode.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(42)

    model_id = "black-forest-labs/FLUX.1-dev"
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    device = torch.device('cuda')
    pipe.to(device)

    height, width, upscale = args.height, args.width, args.upscale
    text_length = 512
    # prompt = "A cat holding a sign that says CVPR 2026"
    prompt = "Mountain landscape with a wooden sign reading “Rectified SpaAttn”."

    control_image = pipe(
        prompt,
        height=height,  
        width=width,
        guidance_scale=3.5,
        num_inference_steps=args.num_steps,
        max_sequence_length=text_length,
        generator=torch.Generator("cuda").manual_seed(0)
    ).images[0]
    pipe.to('cpu')

    # Load pipeline
    controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
    torch_dtype=torch.bfloat16
    )
    pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    pipe.vae.enable_tiling()

    # Upscale x upscale
    width, height = width * upscale, height * upscale
    control_image = control_image.resize((width, height))

    # Jenga
    latent_height_, latent_width_ = height // 16, width // 16
    axis_order_list = [("w","h","t")]
    hilbert_params = build_multi_curve(1, latent_height_, latent_width_, axis_order_list)

    # Prepare sparse params
    img_seq_len = latent_height_ * latent_width_
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
    for k,v in pipe.transformer.attn_processors.items():
        if "single_transformer_blocks" in k:
            attn_processors[k] = RectifiedFluxSpaAttnProcessor2_0(args.mode, select_block_num, hilbert_params[0][2], p_remain_rates, processors_id, text_length)
            processors_id += 1 
        else:
            attn_processors[k] = RectifiedFluxSpaAttnProcessor2_0(args.mode, select_block_num, hilbert_params[0][2], p_remain_rates, processors_id, text_length)
            processors_id += 1  
    pipe.transformer.set_attn_processor(attn_processors)

    torch.cuda.empty_cache()
    with torch.no_grad():
        image = pipe(
            prompt=prompt, 
            control_image=control_image,
            controlnet_conditioning_scale=1.0,
            num_inference_steps=args.num_steps, 
            guidance_scale=3.5,
            height=control_image.size[1],
            width=control_image.size[0],
            max_sequence_length=text_length,
        ).images[0]

    time_flag = datetime.fromtimestamp(time.time()).strftime("%m-%d-%H:%M:%S")
    cur_save_path = f"./{time_flag}_upflux_{va.time_end - va.time_start:.0f}s.png"
    image.save(cur_save_path)
    print(f'Sample save to: {cur_save_path}')




