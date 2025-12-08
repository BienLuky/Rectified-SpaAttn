import torch
import torch.nn.functional as F
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from typing import Any, Dict, Optional, Tuple, Union

from diffusers.models.transformers.transformer_wan import _get_qkv_projections, _get_added_kv_projections
from typing import Optional
from torch.nn.attention import sdpa_kernel, SDPBackend

from .attn import fullattn
from .rectified_wan21_attn import rectified_block_sparse_attention


class RectifiedWanTI2VSpaAttnProcessor2_0:
    def __init__(self, mode, select_block_num, block_neighbor_list, p_remain_rates, processor_id=0, first_frame_blocks=0):
        self.mode = mode
        self.select_block_num = select_block_num
        self.block_neighbor_list = block_neighbor_list
        self.p_remain_rates = p_remain_rates
        self.current_step = 0
        self.processor_id = processor_id
        self.first_frame_blocks = first_frame_blocks

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self._attention_backend = None

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            # key_img = key_img.unflatten(2, (attn.heads, -1))
            # value_img = value_img.unflatten(2, (attn.heads, -1))

            # hidden_states_img = dispatch_attention_fn(
            #     query,
            #     key_img,
            #     value_img,
            #     attn_mask=None,
            #     dropout_p=0.0,
            #     is_causal=False,
            #     backend=self._attention_backend,
            # )
            # hidden_states_img = hidden_states_img.flatten(2, 3)
            # hidden_states_img = hidden_states_img.type_as(query)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states_img = F.scaled_dot_product_attention(
                    query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # hidden_states = dispatch_attention_fn(
        #     query,
        #     key,
        #     value,
        #     attn_mask=attention_mask,
        #     dropout_p=0.0,
        #     is_causal=False,
        #     backend=self._attention_backend,
        # )
        # hidden_states = hidden_states.flatten(2, 3)
        # hidden_states = hidden_states.type_as(query)

        # Attention
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        B, H, S_q, D = query.shape
        B, H, S_k, D = key.shape
        s_k = attention_mask.sum().item() if attention_mask else S_k
        cu_seqlens_q = torch.tensor([0, S_q, S_q], dtype=torch.int32, device=query.device)
        cu_seqlens_kv = torch.tensor([0, s_k, S_k], dtype=torch.int32, device=query.device)
        max_seqlen_q, max_seqlen_kv = S_q, S_k
        if self.mode == "sparse" and (self.processor_id >= 2 and self.current_step >= 10): # Warm up 2 layers and 5 steps
            hidden_states = rectified_block_sparse_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                top_k=self.select_block_num,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                block_neighbor_list=self.block_neighbor_list,
                p_remain_rates=self.p_remain_rates,
                first_frame_blocks=self.first_frame_blocks
            )

        elif self.mode == "sparse":
            hidden_states = fullattn(
                query, key, value, mode="flash", drop_rate=0.0, attn_mask=attention_mask, causal=False, \
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=B)
            hidden_states = hidden_states.transpose(1, 2).reshape(B, S_q, -1)

        elif self.mode in ["flash", "torch", "vanilla"]:
            hidden_states = fullattn(
                query, key, value, mode=self.mode, drop_rate=0.0, attn_mask=attention_mask, causal=False, \
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=B)
            hidden_states = hidden_states.transpose(1, 2).reshape(B, S_q, -1)

        else:
            raise ImportError("Undefined Attention Processor! Just support sparse, flash, torch, vanilla.")

        hidden_states = hidden_states.type_as(query)

        self.current_step += 1
        if self.current_step == 50 * 2:
            self.current_step = 0

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class RectifiedWanT2VSpaAttnProcessor2_0:
    def __init__(self, mode, select_block_num, block_neighbor_list, p_remain_rates, processor_id=0, first_frame_blocks=0, warm_steps=0):
        self.mode = mode
        self.select_block_num = select_block_num
        self.block_neighbor_list = block_neighbor_list
        self.p_remain_rates = p_remain_rates
        self.current_step = 0
        self.processor_id = processor_id
        self.first_frame_blocks = first_frame_blocks
        self.warm_steps = warm_steps

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self._attention_backend = None

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states_img = F.scaled_dot_product_attention(
                    query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # Attention
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        B, H, S_q, D = query.shape
        B, H, S_k, D = key.shape
        s_k = attention_mask.sum().item() if attention_mask else S_k
        cu_seqlens_q = torch.tensor([0, S_q, S_q], dtype=torch.int32, device=query.device)
        cu_seqlens_kv = torch.tensor([0, s_k, S_k], dtype=torch.int32, device=query.device)
        max_seqlen_q, max_seqlen_kv = S_q, S_k
        if self.mode == "sparse" and ((self.processor_id not in [0,1,40,41]) and (self.current_step >= self.warm_steps)): # Warm up 2 layers and 5 steps
            hidden_states = rectified_block_sparse_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                top_k=self.select_block_num,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                block_neighbor_list=self.block_neighbor_list,
                p_remain_rates=self.p_remain_rates,
                first_frame_blocks=self.first_frame_blocks
            )

        elif self.mode == "sparse":
            hidden_states = fullattn(
                query, key, value, mode="flash", drop_rate=0.0, attn_mask=attention_mask, causal=False, \
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=B)
            hidden_states = hidden_states.transpose(1, 2).reshape(B, S_q, -1)

        elif self.mode in ["flash", "torch", "vanilla"]:
            hidden_states = fullattn(
                query, key, value, mode=self.mode, drop_rate=0.0, attn_mask=attention_mask, causal=False, \
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=B)
            hidden_states = hidden_states.transpose(1, 2).reshape(B, S_q, -1)

        else:
            raise ImportError("Undefined Attention Processor! Just support sparse, flash, torch, vanilla.")

        hidden_states = hidden_states.type_as(query)

        self.current_step += 1
        if self.current_step == 40 * 2:
            self.current_step = 0

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class RectifiedWanI2VSpaAttnProcessor2_0:
    def __init__(self, mode, select_block_num, block_neighbor_list, p_remain_rates, processor_id=0, first_frame_blocks=0, warm_steps=0):
        self.mode = mode
        self.select_block_num = select_block_num
        self.block_neighbor_list = block_neighbor_list
        self.p_remain_rates = p_remain_rates
        self.current_step = 0
        self.processor_id = processor_id
        self.first_frame_blocks = first_frame_blocks
        self.warm_steps = warm_steps

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self._attention_backend = None

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states_img = F.scaled_dot_product_attention(
                    query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # Attention
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        B, H, S_q, D = query.shape
        B, H, S_k, D = key.shape
        s_k = attention_mask.sum().item() if attention_mask else S_k
        cu_seqlens_q = torch.tensor([0, S_q, S_q], dtype=torch.int32, device=query.device)
        cu_seqlens_kv = torch.tensor([0, s_k, S_k], dtype=torch.int32, device=query.device)
        max_seqlen_q, max_seqlen_kv = S_q, S_k
        if self.mode == "sparse" and ((self.processor_id not in [0,1,40,41]) and (self.current_step >= self.warm_steps)): # Warm up 2 layers and 5 steps
            hidden_states = rectified_block_sparse_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                top_k=self.select_block_num,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                block_neighbor_list=self.block_neighbor_list,
                p_remain_rates=self.p_remain_rates,
                first_frame_blocks=self.first_frame_blocks
            )

        elif self.mode == "sparse":
            hidden_states = fullattn(
                query, key, value, mode="flash", drop_rate=0.0, attn_mask=attention_mask, causal=False, \
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=B)
            hidden_states = hidden_states.transpose(1, 2).reshape(B, S_q, -1)

        elif self.mode in ["flash", "torch", "vanilla"]:
            hidden_states = fullattn(
                query, key, value, mode=self.mode, drop_rate=0.0, attn_mask=attention_mask, causal=False, \
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=B)
            hidden_states = hidden_states.transpose(1, 2).reshape(B, S_q, -1)

        else:
            raise ImportError("Undefined Attention Processor! Just support sparse, flash, torch, vanilla.")

        hidden_states = hidden_states.type_as(query)

        self.current_step += 1
        if self.current_step == 40 * 2:
            self.current_step = 0

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

