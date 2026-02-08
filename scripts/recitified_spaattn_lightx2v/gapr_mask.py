import torch


def estimate_pr_gain(Q_blocks, K_blocks, q_pools, k_pools, attention_scores):
    """
    Estimates the attention gains and pooling errors of the Pooling Rectification.  
    Q_blocks, K_blocks: (batch, n_heads, blocks-num, block-size, head_dim)  
    q_pools, k_pools: pooling KV (batch, n_heads, blocks-num, head_dim)   
    attention_scores: pooling KV attention acore

    Returns: ~gapr_mask
        gapr_mask: (B, H, NQ, NK) A mask where the pooling correction gain exceeds the pooling error.
    """

    B, H, NQ, IQ, d = Q_blocks.shape
    _, _, NK, JK, _ = K_blocks.shape
    BH = B * H

    delta_q = Q_blocks - q_pools[..., None, :]   # (B, H, NQ, IQ, d)
    delta_k = K_blocks - k_pools[..., None, :]   # (B, H, NK, JK, d)

    # estimate pooling errors
    delta_q_r = delta_q.abs().mean(dim=-2).reshape(BH, NQ, d)    
    k_pools_r = k_pools.reshape(BH, NK, d)         

    dot_q = torch.bmm(delta_q_r, k_pools_r.transpose(-1, -2)) 
    err_q_sum = dot_q.abs() * IQ * JK 

    q_pools_r = q_pools.reshape(BH, NQ, d)       
    delta_k_flat = delta_k.abs().mean(dim=-2).reshape(BH, NK, d) 

    dot_k = torch.bmm(q_pools_r, delta_k_flat.transpose(-1, -2)) 
    err_k_sum = dot_k.abs() * IQ * JK  

    err_score = (err_q_sum + err_k_sum).view(B, H, NQ, NK)  # (B, H, NQ, NK)

    # estimate attention gains
    Gain_score = IQ * JK * attention_scores.abs()

    gapr_mask = Gain_score > err_score

    return ~gapr_mask
