import torch
import triton
import triton.language as tl
from .quant_per_block import per_block_int8


@triton.jit
def _triton_block_sparse_sageattn_fwd_kernel_onehot(
    Q, K, V, Q_scale, K_scale, seqlens, seqlens_q,
    block_mask,  # [B*H, NUM_ROWS, NUM_BLOCKS]
    Out,
    stride_qscaleh, stride_qscalem, stride_qscale1,
    stride_kscaleh, stride_kscalen, stride_kscale1,
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
    q_scale_ptr = Q_scale + off_hz * stride_qscaleh + start_m
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    k_scale_ptr = K_scale + off_hz * stride_kscaleh
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # mask row: [B*H, M_BLOCKS, N_BLOCKS]
    mask_ptr = block_mask + off_hz * stride_bh + start_m * stride_bm

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q) # , mask=offs_m[:, None] < seqlen, other=0.0
    q_scale = tl.load(q_scale_ptr)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen_q

    for block_idx in range(NUM_BLOCKS):
        is_valid_block = tl.load(mask_ptr + block_idx * stride_bn)

        if is_valid_block:
            k_scale = tl.load(k_scale_ptr + block_idx)
            start_n = block_idx * BLOCK_N
            cols = start_n + offs_n
            
            # -- load k, v --
            k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=cols[None, :] < seqlen)
            v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=cols[:, None] < seqlen)
            
            # -- compute qk --
            qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)
            qk = tl.where(m_mask, qk, float("-inf"))


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
            p = p.to(tl.float16)
            acc += tl.dot(p, v, out_dtype=tl.float16)
            
            # -- update m_i and l_i --
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_block_sparse_sageattention_onehot(
    q,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens,           # [BATCH, ]
    block_mask,        # [BATCH, N_HEADS, NUM_QUERIES, NUM_BLOCKS] one-hot boolean mask
    sm_scale,
    block_size_M=128,
    block_size_N=128,
    tensor_layout="HND",
    smooth_k=True,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    batch_size, n_heads, S, D = q.shape
    seqlens_q = seqlens.clone()

    # SageAttention
    assert q.is_cuda, "Input tensors must be on cuda."
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32], "Input tensors must be in dtype of torch.float16, torch.bfloat16, or torch.float32."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    headdim = q.size(-1)
    assert headdim in [64, 96, 128], "headdim should be in [64, 96, 128]."
    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    seq_dim = 1 if tensor_layout == "NHD" else 2
    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        k -= km
    else:
        km = None
    if q.dtype == torch.bfloat16 or q.dtype == torch.float32:
        v = v.to(torch.float16)

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, BLKQ=block_size_M, BLKK=block_size_N, sm_scale=sm_scale, tensor_layout=tensor_layout)

    q_int8 = q_int8.reshape(-1, q_int8.shape[2], q_int8.shape[3]).contiguous()
    k_int8 = k_int8.reshape(-1, k_int8.shape[2], k_int8.shape[3]).contiguous()
    q_scale = q_scale.reshape(-1, q_scale.shape[2], q_scale.shape[3]).contiguous()
    k_scale = k_scale.reshape(-1, k_scale.shape[2], k_scale.shape[3]).contiguous()
    v = v.reshape(-1, v.shape[2], v.shape[3]).contiguous()
    o = torch.zeros_like(q_int8, dtype=q.dtype)

    num_query_blocks = block_mask.shape[-2]
    num_blocks = block_mask.shape[-1]
    
    block_mask_reshaped = block_mask.reshape(batch_size * n_heads, num_query_blocks, num_blocks)

    grid = (num_query_blocks, batch_size * n_heads, 1)

    if q.dtype == torch.bfloat16:
        dtype = tl.bfloat16
    else:
        dtype = tl.float16

    if not seqlens.device == q.device:
        seqlens = seqlens.to(q.device)
    if not block_mask_reshaped.device == q.device:
        block_mask_reshaped = block_mask_reshaped.to(q.device)

    with torch.cuda.device(q.device):
        _triton_block_sparse_sageattn_fwd_kernel_onehot[grid](
            q_int8, k_int8, v, q_scale, k_scale, seqlens, seqlens_q,
            block_mask_reshaped,
            o,
            q_scale.stride(0), q_scale.stride(1), q_scale.stride(2), 
            k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), 
            q_int8.stride(0), q_int8.stride(1), q_int8.stride(2), 
            k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), 
            v.stride(0), v.stride(1), v.stride(2), 
            o.stride(0), o.stride(1), o.stride(2), 
            block_mask_reshaped.stride(0), block_mask_reshaped.stride(1), block_mask_reshaped.stride(2),
            q_int8.shape[0], q_int8.shape[1], 
            num_blocks,
            BLOCK_M=block_size_M, BLOCK_N=block_size_N,
            BLOCK_DMODEL=Lk,
            dtype=dtype,
        )
    return o.reshape(batch_size, n_heads, S, D)
