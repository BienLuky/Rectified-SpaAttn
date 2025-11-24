import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from diffusers.models.attention_processor import Attention
from typing import Optional
from torch.nn.attention import sdpa_kernel, SDPBackend

from .attn import fullattn
from .gapr_mask import estimate_pr_gain


@triton.jit
def _triton_block_sparse_attn_fwd_kernel_onehot(
    Q, K, V, seqlens, seqlens_q, qk_scale,
    block_mask,  # [B*H, NUM_ROWS, NUM_BLOCKS]
    Out,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_om, stride_ok,
    stride_bh, stride_bm, stride_bn,
    H, N_CTX,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)  # current query block
    off_hz = tl.program_id(1)   # B*H index

    seqlen = tl.load(seqlens + off_hz // H)
    seqlen_q = tl.load(seqlens_q + off_hz // H)
    if start_m * BLOCK_M >= seqlen_q:
        return

    # offset
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # mask row: [B*H, M_BLOCKS, N_BLOCKS]
    mask_ptr = block_mask + off_hz * stride_bh + start_m * stride_bm

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q) # , mask=offs_m[:, None] < seqlen, other=0.0
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen_q

    for block_idx in range(NUM_BLOCKS):
        is_valid_block = tl.load(mask_ptr + block_idx * stride_bn)

        if is_valid_block:
            start_n = block_idx * BLOCK_N
            cols = start_n + offs_n
            
            # -- load k, v --
            k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=cols[None, :] < seqlen, other=0.0)
            v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=cols[:, None] < seqlen, other=0.0)
            
            # -- compute qk --
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            
            # Safer way to limit KV: use original m_mask, then apply kv range check after qk matrix calculation
            qk = tl.where(m_mask, qk, float("-inf"))
            qk += tl.dot(q, k)

            # Create KV mask and apply
            kv_valid = cols[None, :] < seqlen
            qk = tl.where(kv_valid, qk, float("-inf"))
            
            # -- compute scaling constant --
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2(m_i - m_i_new)
            p = tl.math.exp2(qk - m_i_new[:, None])
            
            # -- scale and update acc --
            acc_scale = l_i * 0 + alpha  # workaround some compiler bug
            acc *= acc_scale[:, None]
            acc += tl.dot(p.to(dtype), v)
            
            # -- update m_i and l_i --
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_block_sparse_attention_onehot(
    q,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens,           # [BATCH, ]
    block_mask,        # [BATCH, N_HEADS, NUM_QUERIES, NUM_BLOCKS] one-hot boolean mask
    sm_scale,
    block_size_M=128,
    block_size_N=128,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    batch_size, n_heads, S, D = q.shape
    seqlens_q = seqlens.clone()

    # [B*H, S, D]
    q = q.reshape(-1, q.shape[2], q.shape[3]).contiguous()
    k = k.reshape(-1, k.shape[2], k.shape[3]).contiguous()
    v = v.reshape(-1, v.shape[2], v.shape[3]).contiguous()
    o = torch.zeros_like(q)

    num_query_blocks = block_mask.shape[-2]
    num_blocks = block_mask.shape[-1]
    
    # [batch*heads, queries, blocks] for adapting triton kernel
    block_mask_reshaped = block_mask.reshape(batch_size * n_heads, num_query_blocks, num_blocks)

    grid = (num_query_blocks, batch_size * n_heads, 1)
    
    if q.dtype == torch.bfloat16:
        dtype = tl.bfloat16
    else:
        dtype = tl.float16

    qk_scale = sm_scale * 1.44269504

    if not seqlens.device == q.device:
        seqlens = seqlens.to(q.device)
    if not block_mask_reshaped.device == q.device:
        block_mask_reshaped = block_mask_reshaped.to(q.device)

    with torch.cuda.device(q.device):
        _triton_block_sparse_attn_fwd_kernel_onehot[grid](
            q, k, v, seqlens, seqlens_q, qk_scale,
            block_mask_reshaped,
            o,
            q.stride(0), q.stride(1), q.stride(2), 
            k.stride(0), k.stride(1), k.stride(2), 
            v.stride(0), v.stride(1), v.stride(2), 
            o.stride(0), o.stride(1), o.stride(2), 
            block_mask_reshaped.stride(0), block_mask_reshaped.stride(1), block_mask_reshaped.stride(2),
            q.shape[0], q.shape[1], 
            num_blocks,
            BLOCK_M=block_size_M, BLOCK_N=block_size_N,
            BLOCK_DMODEL=Lk,
            dtype=dtype,
        )
    return o.reshape(batch_size, n_heads, S, D)


def _build_block_index_with_importance_optimized(
    query: torch.Tensor,     # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,       # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 128,
    block_size_N: int = 128,
    text_start_block: int = None,  
    text_end_block: int = None,  
    num_blocks: int = None,        
    prob_threshold: float = 0.7,   
    block_neighbor_list: torch.Tensor = None,  # [block_num, block_num] one-hot tensor
    first_frame_blocks: int = None,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    num_query_blocks = (context_size + block_size_M - 1) // block_size_M
    device = query.device

    # 1. Pool visual queries and keys
    Q_blocks = query.reshape((batch_size, num_heads, -1, block_size_M, head_dim))
    query_pool = Q_blocks.mean(dim=-2)
    K_blocks = key.reshape((batch_size, num_heads, -1, block_size_N, head_dim))
    key_pool = K_blocks.mean(dim=-2)

    # 2. Calculate attention scores - using bmm optimization
    # Reshape to [batch_size * num_heads, num_query_blocks, head_dim]
    q_bmm = query_pool.reshape(batch_size * num_heads, query_pool.shape[2], head_dim)
    
    # Reshape to [batch_size * num_heads, head_dim, num_key_blocks]
    k_bmm = key_pool.reshape(batch_size * num_heads, key_pool.shape[2], head_dim).transpose(1, 2)
    
    # 3. Process scores for visual blocks
    # Use bmm for batch matrix multiplication
    attention_scores_flat = torch.bmm(q_bmm, k_bmm).reshape(batch_size, num_heads, query_pool.shape[2], key_pool.shape[2])

    # Reshape back to original dimensions [batch_size, num_heads, num_query_blocks, num_key_blocks]
    attention_scores = attention_scores_flat * (head_dim ** -0.5)

    # Only process scores for non-text blocks
    normal_scores = attention_scores[:, :, :, :text_start_block]
    # normal_scores = attention_scores[:, :, :, :text_end_block]
    
    # 4. Use direct softmax to calculate probability distribution for each query
    probs = torch.softmax(normal_scores, dim=-1)

    # 5. GAPR to calculate rectified noncritical blocks mask
    nogapr_mask = None
    nogapr_mask = estimate_pr_gain(Q_blocks, K_blocks, query_pool, key_pool, attention_scores_flat)

    # 6. Sort probability distribution for each head and query
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # 7. Find number of blocks needed for each (batch, head, query) position
    mask = cumsum_probs <= prob_threshold
    num_blocks_needed = mask.sum(dim=-1) + 1  # [batch, heads, queries]
    num_blocks_needed = torch.maximum(
        num_blocks_needed,
        torch.tensor(top_k, device=device)
    )
    
    # Create one-hot output tensor [batch_size, num_heads, num_query_blocks, num_blocks]
    one_hot_output = torch.zeros(
        (batch_size, num_heads, num_query_blocks, num_blocks), 
        dtype=torch.bool, device=device
    )

    max_k = indices.shape[-1]
    # Use einsum-based indexing for reduced memory:
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1, 1).expand(-1, num_heads, num_query_blocks, max_k)
    head_idx = torch.arange(num_heads, device=device).view(1, -1, 1, 1).expand(batch_size, -1, num_query_blocks, max_k)
    query_idx = torch.arange(num_query_blocks, device=device).view(1, 1, -1, 1).expand(batch_size, num_heads, -1, max_k)
    k_idx = torch.arange(max_k, device=device).view(1, 1, 1, -1).expand(batch_size, num_heads, num_query_blocks, -1)

    # Create mask more efficiently
    valid_mask = k_idx < num_blocks_needed.unsqueeze(-1)
    
    # Find all positions that need to be filled
    b_indices = batch_idx[valid_mask]
    h_indices = head_idx[valid_mask]
    q_indices = query_idx[valid_mask]

    # Get index values corresponding to these positions
    flat_indices = indices[b_indices, h_indices, q_indices, k_idx[valid_mask]]
    
    # Use scatter and index operations to fill in one go
    one_hot_output[b_indices, h_indices, q_indices, flat_indices] = True
    
    # Add physical neighbors - directly take union
    if block_neighbor_list is not None:
        # Ensure block_neighbor_list is on the correct device
        if block_neighbor_list.device != device:
            block_neighbor_list = block_neighbor_list.to(device)
        
        # Ensure dimensions match and convert to boolean
        neighbor_mask = block_neighbor_list[:num_query_blocks, :text_start_block].bool()
        
        # Expand to [batch, heads, q_blocks, blocks] dimension and take union with existing output
        one_hot_output[:, :, :neighbor_mask.shape[0], :text_start_block] |= neighbor_mask.unsqueeze(0).unsqueeze(0)
    
    if first_frame_blocks > 0:
        one_hot_output[:, :, :first_frame_blocks, :first_frame_blocks] = True

    return one_hot_output, probs, nogapr_mask


def block_sparse_attention_combined(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    attn_mask: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, N_CTX]
    top_k: int,
    block_size_M: int = 128,
    block_size_N: int = 128,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_kv: torch.Tensor = None,
    max_seqlen_q: int = None,
    max_seqlen_kv: int = None,
    prob_threshold: float = 0.5,  # new parameter
    block_neighbor_list: torch.Tensor = None,
    shape_xfuse: bool = False,
    first_frame_blocks: int = None,
):
    """
    Combined attention processing for visual blocks:
    1. Visual blocks select top-k blocks based on importance (without causal constraints)
    """
    batch_size, num_heads, context_size, head_dim = query.shape
    
    pad = block_size_M - (context_size % block_size_M) if context_size % block_size_M != 0 else 0
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])
    seqlens = torch.tensor([context_size] * batch_size, dtype=torch.int32).to(
                device=query.device, non_blocking=True)
    
    sm_scale = head_dim ** -0.5
    padded_context_size = query.shape[2]
    num_blocks = (padded_context_size + block_size_M - 1) // block_size_M
    
    # Compute normal_blocks, normal_tokens only once
    normal_blocks = num_blocks
    text_end_block = (seqlens.data + block_size_N - 1) // block_size_N
    normal_tokens = normal_blocks * block_size_M
    
    # 1. process visual blocks (sparse attention to visual blocks)
    if normal_blocks > 0:
        query_normal = query[:, :, :normal_tokens, :]
            
        # Pass pre-computed pools to block index function
        block_relation_onehot, probs, nogapr_mask = _build_block_index_with_importance_optimized(
            query_normal, key, top_k, block_size_M, block_size_N, 
            text_start_block=normal_blocks, text_end_block=text_end_block, num_blocks=num_blocks,
            prob_threshold=prob_threshold,
            block_neighbor_list=block_neighbor_list,
            first_frame_blocks=first_frame_blocks,
        )

        # Rectifying the Attention Bias of Critical Tokens 
        one_hot_output_partical = block_relation_onehot[:, :, :, :text_end_block].clone()
        one_hot_output_partical[:, :, :, :normal_blocks] |= nogapr_mask
        attn_pool = probs.masked_fill(~(one_hot_output_partical), 0.0)
        attn_pool_sum = torch.sum(attn_pool, dim=-1)
        rectified_factor_R = attn_pool_sum.repeat_interleave(block_size_M, dim=-1) # [B, H, L_q]

        # Rectifying the Attention Bias of Non-Critical Tokens 
        attn_pool_novalid = probs.masked_fill(one_hot_output_partical, 0.0)
        value_pool = value.reshape((batch_size, num_heads, -1, block_size_N, head_dim)).mean(dim=-2)[:, :, :text_end_block, :]
        rectified_noncriattention = torch.matmul(attn_pool_novalid, value_pool).repeat_interleave(block_size_M, dim=-2)

        # direct use one-hot version sparse attention
        output_normal = _triton_block_sparse_attention_onehot(
            query_normal, key, value, seqlens, 
            block_relation_onehot, sm_scale, block_size_M, block_size_N,
        )

        output_normal = output_normal * rectified_factor_R.to(query.dtype).unsqueeze(-1) + rectified_noncriattention
    else:
        output_normal = torch.empty(0, device=query.device)

    # merge outputs
    output = output = output_normal[:, :, :context_size, :]
    
    if not shape_xfuse:
        output = output.permute(0, 2, 1, 3).reshape(batch_size, context_size, -1)
        return output
    # remove padding
    return output.permute(0, 2, 1, 3)


# keep the original function as an alias for backward compatibility
def rectified_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,     
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    top_k: int,
    block_size_M: int = 128,
    block_size_N: int = 128,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_kv: torch.Tensor = None,
    max_seqlen_q: int = None,
    max_seqlen_kv: int = None,
    block_neighbor_list: torch.Tensor = None,
    shape_xfuse: bool = False,
    p_remain_rates: float = 0.5,
    first_frame_blocks: int = None,
):
    """
    backward compatible wrapper around block_sparse_attention_combined.
    """
    return block_sparse_attention_combined(
        query, key, value, attn_mask, top_k, block_size_M, block_size_N,
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, 
        block_neighbor_list=block_neighbor_list, shape_xfuse=shape_xfuse,
        prob_threshold=p_remain_rates, first_frame_blocks=first_frame_blocks
    )


class RectifiedWanT2VSpaAttnProcessor2_0:
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

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states_img = F.scaled_dot_product_attention(
                    query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # Attention
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


class RectifiedWanI2VSpaAttnProcessor2_0:
    def __init__(self, 
            mode, select_block_num, block_neighbor_list, p_remain_rates, processor_id=0, first_frame_blocks=0):
        self.mode = mode
        self.select_block_num = select_block_num
        self.block_neighbor_list = block_neighbor_list
        self.p_remain_rates = p_remain_rates
        self.current_step = 0
        self.processor_id = processor_id
        self.first_frame_blocks = first_frame_blocks

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states_img = F.scaled_dot_product_attention(
                    query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # 5. Attention
        B, H, S_q, D = query.shape
        B, H, S_k, D = key.shape
        s_k = attention_mask.sum().item() if attention_mask else S_k
        cu_seqlens_q = torch.tensor([0, S_q, S_q], dtype=torch.int32, device=query.device)
        cu_seqlens_kv = torch.tensor([0, s_k, S_k], dtype=torch.int32, device=query.device)
        max_seqlen_q, max_seqlen_kv = S_q, S_k
        if self.mode == "sparse" and self.processor_id >= 2: # Warm up 2 layers
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
