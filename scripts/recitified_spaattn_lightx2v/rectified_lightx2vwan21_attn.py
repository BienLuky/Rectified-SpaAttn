import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import sys
import os
sys.path.append(os.path.dirname(__file__))

from .gapr_mask import estimate_pr_gain
from .spaattn_fwd_triton import _triton_block_sparse_attention_onehot
from .spasageattn_fwd_triton import _triton_block_sparse_sageattention_onehot


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
    sage: bool = False,
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
        if sage:
            output_normal = _triton_block_sparse_sageattention_onehot(
                query_normal, key, value, seqlens, 
                block_relation_onehot, sm_scale, block_size_M, block_size_N,
            )
        else:
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
    sage: bool = False,
):
    """
    backward compatible wrapper around block_sparse_attention_combined.
    """
    return block_sparse_attention_combined(
        query, key, value, attn_mask, top_k, block_size_M, block_size_N,
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, 
        block_neighbor_list=block_neighbor_list, shape_xfuse=shape_xfuse,
        prob_threshold=p_remain_rates, first_frame_blocks=first_frame_blocks, sage=sage
    )


def RectifiedLightX2VWanSpaAttnProcessor2_0(query, key, value, attention_mask, select_block_num, block_neighbor_list, p_remain_rates, first_frame_blocks):

    # Attention
    B, H, S_q, D = query.shape
    B, H, S_k, D = key.shape
    s_k = attention_mask.sum().item() if attention_mask else S_k
    cu_seqlens_q = torch.tensor([0, S_q, S_q], dtype=torch.int32, device=query.device)
    cu_seqlens_kv = torch.tensor([0, s_k, S_k], dtype=torch.int32, device=query.device)
    max_seqlen_q, max_seqlen_kv = S_q, S_k

    hidden_states = rectified_block_sparse_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        top_k=select_block_num,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        block_neighbor_list=block_neighbor_list,
        p_remain_rates=p_remain_rates,
        first_frame_blocks=first_frame_blocks,
        sage=True
    )

    return hidden_states

