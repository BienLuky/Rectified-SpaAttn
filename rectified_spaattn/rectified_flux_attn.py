import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from diffusers.models.attention_processor import Attention
from typing import Optional

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
    seqlens_q = torch.full_like(seqlens, S, dtype=torch.int32)

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
    attenable: int = None,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    num_query_blocks = (context_size + block_size_M - 1) // block_size_M
    device = query.device

    # 1. Pool visual queries and keys, isolated text keys
    Q_blocks = query.reshape((batch_size, num_heads, -1, block_size_M, head_dim))
    query_pool = Q_blocks.mean(dim=-2)
    K_blocks = key[:, :, :num_query_blocks*block_size_N, :].reshape((batch_size, num_heads, -1, block_size_N, head_dim))
    key_pool_normal = K_blocks.mean(dim=-2)
    key_text = key[:, :, num_query_blocks*block_size_N : num_query_blocks*block_size_N+attenable, :]
    key_pool = torch.cat((key_pool_normal, key_text), dim=-2)

    # 2. Calculate attention scores - using bmm optimization
    # Reshape to [batch_size * num_heads, num_query_blocks, head_dim]
    q_bmm = query_pool.reshape(batch_size * num_heads, query_pool.shape[2], head_dim)
    
    # Reshape to [batch_size * num_heads, head_dim, num_key_blocks]
    k_bmm = key_pool.reshape(batch_size * num_heads, key_pool.shape[2], head_dim).transpose(1, 2)
    
    # 3. Process scores for visual blocks and text tokens
    # Use bmm for batch matrix multiplication
    attention_scores_flat = torch.bmm(q_bmm, k_bmm).reshape(batch_size, num_heads, query_pool.shape[2], key_pool.shape[2])

    # Reshape back to original dimensions [batch_size, num_heads, num_query_blocks, num_key_blocks]
    attention_scores = attention_scores_flat * (head_dim ** -0.5)

    # 4. Use direct softmax to calculate probability distribution for each query
    probs = torch.softmax(attention_scores, dim=-1)

    # 5. GAPR to calculate rectified noncritical blocks mask
    nogapr_mask = None
    nogapr_mask = estimate_pr_gain(Q_blocks, K_blocks, query_pool, key_pool_normal, attention_scores_flat[..., :-attenable])

    # 6. IPAR to calculate accurate implicit full attention (Attention Reallocation)
    normal_probs = probs[:, :, :, :num_query_blocks]
    normal_sum = normal_probs.sum(dim=3, keepdim=True)
    text_probs = probs[:, :, :, num_query_blocks:].sum(dim=3, keepdim=True)
    normal_probs_gt = normal_probs * block_size_N / (normal_sum * block_size_N + text_probs)
    text_probs_gt = text_probs / (normal_sum * block_size_N + text_probs)
    probs = torch.cat((normal_probs_gt, text_probs_gt), dim=-1)

    # 7. Sort probability distribution for each head and query
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # 8. Find number of blocks needed for each (batch, head, query) position
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
    
    # # Add physical neighbors - directly take union
    if block_neighbor_list is not None:
        # Ensure block_neighbor_list is on the correct device
        if block_neighbor_list.device != device:
            block_neighbor_list = block_neighbor_list.to(device)
        
        # Ensure dimensions match and convert to boolean
        neighbor_mask = block_neighbor_list[:num_query_blocks, :text_start_block].bool()
        
        # Expand to [batch, heads, q_blocks, blocks] dimension and take union with existing output
        one_hot_output[:, :, :neighbor_mask.shape[0], :text_start_block] |= neighbor_mask.unsqueeze(0).unsqueeze(0)
    
    # # Add text blocks - all batches, all heads, all query blocks can see all text blocks
    one_hot_output[:, :, :, text_start_block:text_end_block] = True

    # Return sparse mask, implicit full attention, GAPR mask
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
    text_length: int = 256,
    shape_xfuse: bool = False,
):
    """
    Combined attention processing for normal blocks and text blocks:
    1. Normal blocks select top-k blocks based on importance (without causal constraints)
    2. Text blocks get full attention (can see all blocks)
    3. All normal blocks can see all text blocks
    """
    batch_size, num_heads, context_size, head_dim = query.shape
    
    seqlens = cu_seqlens_kv[1:2]
    attenable = text_length
    seqlens = seqlens.to(torch.int32).to(query.device)

    sm_scale = head_dim ** -0.5
    padded_context_size = query.shape[2]
    num_blocks = (padded_context_size + block_size_M - 1) // block_size_M
    
    # Compute normal_blocks, normal_tokens only once
    normal_blocks = num_blocks - (text_length // block_size_M)
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
            attenable=attenable
        )

        # Rectifying the Attention Bias of Critical Tokens 
        one_hot_output_partical = block_relation_onehot[:, :, :, :normal_blocks+1].clone()
        one_hot_output_partical[:, :, :, :normal_blocks] |= nogapr_mask
        attn_pool = probs.masked_fill(~(one_hot_output_partical), 0.0)
        attn_pool_sum = torch.sum(attn_pool, dim=-1)
        rectified_factor_R = attn_pool_sum.repeat_interleave(block_size_M, dim=-1) # [B, H, L_q]

        # Rectifying the Attention Bias of Non-Critical Tokens 
        attn_pool_novalid = probs.masked_fill(one_hot_output_partical, 0.0)
        value_pool = value.reshape((batch_size, num_heads, -1, block_size_N, head_dim)).mean(dim=-2)[:, :, :normal_blocks+1, :]
        rectified_noncriattention = torch.matmul(attn_pool_novalid, value_pool).repeat_interleave(block_size_M, dim=-2)

        # direct use one-hot version sparse attention
        output_normal = _triton_block_sparse_attention_onehot(
            query_normal, key, value, seqlens, 
            block_relation_onehot, sm_scale, block_size_M, block_size_N,
        )

        output_normal = output_normal * rectified_factor_R.unsqueeze(-1) + rectified_noncriattention
    else:
        output_normal = torch.empty(0, device=query.device)

    # 2. process text blocks (full attention to all blocks)
    # extract text blocks
    query_text = query[:, :, normal_tokens:, :]
    key_text = key  # can see all keys
    value_text = value
    # use Flash Attention
    cu_seqlens_q_text = cu_seqlens_q.clone()
    cu_seqlens_q_text[1:] -= normal_tokens
    max_seqlen_q_text = max_seqlen_q - normal_tokens
    # attn_mask_text = attn_mask[:, :, :, normal_tokens:]
    output_text = fullattn(
        query_text, key_text, value_text, "flash", drop_rate=0.0, attn_mask=attn_mask, causal=False, \
        cu_seqlens_q=cu_seqlens_q_text, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q_text, max_seqlen_kv=max_seqlen_kv, batch_size=batch_size)

    # merge outputs
    output = torch.cat([output_normal, output_text], dim=2)
    
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
    text_length: int = 256,
):
    """
    backward compatible wrapper around block_sparse_attention_combined.
    """
    return block_sparse_attention_combined(
        query, key, value, attn_mask, top_k, block_size_M, block_size_N,
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, 
        block_neighbor_list=block_neighbor_list, shape_xfuse=shape_xfuse,
        prob_threshold=p_remain_rates, text_length=text_length
    )


class RectifiedFluxSpaAttnProcessor2_0:
    def __init__(self, mode, select_block_num, block_neighbor_list, p_remain_rates, processor_id=0, text_length=256):
        self.mode = mode
        self.select_block_num = select_block_num
        self.block_neighbor_list = block_neighbor_list
        self.p_remain_rates = p_remain_rates
        self.current_step = 0
        self.processor_id = processor_id
        self.text_length = text_length

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # # attention
            # query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            # key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            # value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            # Jenga attention
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Attention
        B, H, S_q, D = query.shape
        B, H, S_k, D = key.shape
        s_k = attention_mask.sum().item() if attention_mask else S_k
        cu_seqlens_q = torch.tensor([0, S_q, S_q], dtype=torch.int32, device=query.device)
        cu_seqlens_kv = torch.tensor([0, s_k, S_k], dtype=torch.int32, device=query.device)
        max_seqlen_q, max_seqlen_kv = S_q, S_k
        if self.mode == "sparse" and (self.processor_id<37 or self.processor_id>=57): # warmup 37-56 layers
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
                text_length=self.text_length
            )

        else:
            hidden_states = fullattn(
                query, key, value, mode="flash", drop_rate=0.0, attn_mask=attention_mask, causal=False, \
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, batch_size=B)
            hidden_states = hidden_states.transpose(1, 2).reshape(B, S_q, -1)

        hidden_states = hidden_states.to(query.dtype)

        self.current_step += 1
        if self.current_step == 50:
            self.current_step = 0
            
        if encoder_hidden_states is not None:
            # encoder_hidden_states, hidden_states = (
            #     hidden_states[:, : encoder_hidden_states.shape[1]],
            #     hidden_states[:, encoder_hidden_states.shape[1] :],
            # )

            # Jenga
            hidden_states, encoder_hidden_states = (
                hidden_states[:, :-encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1]:],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
