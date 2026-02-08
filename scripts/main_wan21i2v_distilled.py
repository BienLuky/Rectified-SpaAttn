"""
Wan2.2 distilled model image-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.2 distilled model for I2V generation.
"""

import argparse, math, torch, types
from lightx2v import LightX2VPipeline
from recitified_spaattn_lightx2v.jenga_reorder import build_multi_curve
from recitified_spaattn_lightx2v.rectified_lightx2vwan21_attn import RectifiedLightX2VWanSpaAttnProcessor2_0


def infer_main_blocks_jenga(self, blocks, pre_infer_out):

    # jenga 
    hidden_states = pre_infer_out.x
    hidden_states = hidden_states[self.hilbert_order]
    pre_infer_out.x = hidden_states
    self.cos_sin = self.cos_sin[self.hilbert_order]

    x = self.infer_func(blocks, pre_infer_out.x, pre_infer_out)

    # Jenga
    x = x[self.linear_to_hilbert]

    return x


def infer_self_attn_rectified_spaattn(self, phase, x, shift_msa, scale_msa):
    cos_sin = self.cos_sin
    if hasattr(phase, "smooth_norm1_weight"):
        norm1_weight = (1 + scale_msa.squeeze()) * phase.smooth_norm1_weight.tensor
        norm1_bias = shift_msa.squeeze() * phase.smooth_norm1_bias.tensor
        norm1_out = phase.norm1.apply(x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.sensitive_layer_dtype)
        norm1_out.mul_(norm1_weight).add_(norm1_bias)
    else:
        norm1_out = phase.norm1.apply(x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.sensitive_layer_dtype)
        norm1_out = self.modulate_func(norm1_out, scale=scale_msa, shift=shift_msa).squeeze()

    if self.sensitive_layer_dtype != self.infer_dtype:
        norm1_out = norm1_out.to(self.infer_dtype)

    s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim

    q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(s, n, d)
    k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(s, n, d)
    v = phase.self_attn_v.apply(norm1_out).view(s, n, d)

    q, k = self.apply_rope_func(q, k, cos_sin)

    img_qkv_len = q.shape[0]
    if self.self_attn_cu_seqlens_qkv is None:
        if self.self_attn_1_type in ["flash_attn2", "flash_attn3"]:
            self.self_attn_cu_seqlens_qkv = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)
        else:
            self.self_attn_cu_seqlens_qkv = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32)

    if self.clean_cuda_cache:
        del norm1_out, shift_msa, scale_msa
        torch.cuda.empty_cache()

    attn_running_args = {
        "block_idx": self.block_idx,
        "scheduler": self.scheduler,
    }

    if self.config["seq_parallel"]:
        attn_out = phase.self_attn_1_parallel.apply(
            q=q,
            k=k,
            v=v,
            slice_qkv_len=img_qkv_len,
            cu_seqlens_qkv=self.self_attn_cu_seqlens_qkv,
            attention_module=phase.self_attn_1,
            attention_type=self.self_attn_1_type,
            seq_p_group=self.seq_p_group,
            use_fp8_comm=self.seq_p_fp8_comm,
            use_tensor_fusion=self.seq_p_tensor_fusion,
            enable_head_parallel=self.enable_head_parallel,
            **attn_running_args,
        )
    else:
        # sparse
        q = q.permute(1, 0, 2).unsqueeze(0)
        k = k.permute(1, 0, 2).unsqueeze(0)
        v = v.permute(1, 0, 2).unsqueeze(0)
        attn_out = RectifiedLightX2VWanSpaAttnProcessor2_0(q, k, v, None, self.select_block_num, self.block_neighbor_list, self.p_remain_rates, self.first_frame_blocks)
        attn_out = attn_out.squeeze(0)

    y = phase.self_attn_o.apply(attn_out)

    if self.clean_cuda_cache:
        del q, k, v, attn_out
        torch.cuda.empty_cache()

    return y
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=720, help="video height.")
    parser.add_argument("--width", type=int, default=1280, help="video width.")
    parser.add_argument("--frame", type=int, default=81, help="video frames.")
    parser.add_argument("--sa_drop_rate", type=float, default=0.75, help="Drop rate for sparse attention.")
    parser.add_argument("--p_remain_rates", type=float, default=0.3, help="Probability for keeping tokens.")
    parser.add_argument("--mode", type=str, default="sparse", choices=["sparse", "flash", "torch", "vanilla"], help="Inference mode.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    path = "./models/wan2.1_i2v_720p"
    pipe = LightX2VPipeline(
        model_path="./models/Wan2.1-I2V-14B-720P",
        model_cls="wan2.1_distill",
        task="i2v",
        dit_original_ckpt=f"{path}/wan2.1_i2v_720p_lightx2v_4step.safetensors",
    )

    # Create generator manually with specified parameters
    pipe.create_generator(
        infer_steps=4,
        height=720,  # Can be set to 720 for higher resolution
        width=1280,  # Can be set to 1280 for higher resolution
        num_frames=81,
        guidance_scale=1,
        sample_shift=5.0,
    )

    seed = 42
    prompt = "a bald eagle flying over a tree filled forest."
    negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    save_result_path = "./output_i2v.mp4"
    image_path = "LightX2V/assets/a bald eagle flying over a tree filled forest.jpg"

    height, width, frame = 704, args.width, args.frame
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

    # # Jenga
    pipe.runner.model.transformer_infer.infer_main_blocks = types.MethodType(
                                                            infer_main_blocks_jenga,
                                                            pipe.runner.model.transformer_infer
                                                            )
    pipe.runner.model.transformer_infer.infer_self_attn = types.MethodType(
                                                            infer_self_attn_rectified_spaattn,
                                                            pipe.runner.model.transformer_infer
                                                            )
    
    pipe.runner.model.transformer_infer.linear_to_hilbert = hilbert_params[0][0]
    pipe.runner.model.transformer_infer.hilbert_order = hilbert_params[0][1]
    pipe.runner.model.transformer_infer.block_neighbor_list = hilbert_params[0][2]
    pipe.runner.model.transformer_infer.select_block_num = select_block_num
    pipe.runner.model.transformer_infer.p_remain_rates = p_remain_rates
    pipe.runner.model.transformer_infer.first_frame_blocks = first_frame_blocks

    pipe.generate(
        seed=seed,
        image_path=image_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        save_result_path=save_result_path,
    )
