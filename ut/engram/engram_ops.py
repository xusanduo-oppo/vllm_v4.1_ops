import torch

@torch.no_grad()
def hash_ids_torch(
    input_ids, token_map, dead_mask, positions, block_table, query_start_loc,
    multipliers, primes, offsets, cache, lookback_token_ids, lookback_dead_mask,
    pad_id: int, dead_id: int = -1, cache_block_size: int = 32,
):
    T = input_ids.shape[0]
    L, MAX_NGRAM = multipliers.shape
    H = primes.shape[-1]
    C = (MAX_NGRAM - 1) * H
    device = input_ids.device
    num_req = query_start_loc.numel() - 1
    lookback_depth = lookback_token_ids.shape[1]
    num_slots = cache.shape[0] if cache is not None else 0
    num_table_rows = block_table.shape[0] if block_table is not None else 0
    max_blocks = block_table.shape[1] if block_table is not None else 0

    if T == 0:
        return torch.empty((0, L, C), dtype=torch.int32, device=device)

    token_idx = torch.arange(T, device=device, dtype=query_start_loc.dtype)
    req = torch.searchsorted(query_start_loc, token_idx, right=True) - 1
    req = req.clamp(0, num_req - 1)
    chunk_idx = query_start_loc[req].to(torch.long).clamp(max=T - 1)
    chunk_start = positions[chunk_idx]
    position = positions

    rolling = torch.zeros((T, L), dtype=torch.int64, device=device)
    blocked = torch.zeros(T, dtype=torch.bool, device=device)
    pad_tensor = torch.full((T,), pad_id, dtype=torch.int64, device=device)
    output = torch.empty((T, L, C), dtype=torch.int32, device=device)

    for shift in range(MAX_NGRAM):
        lookback = position - shift
        in_batch = lookback >= chunk_start

        # 分支 A
        batch_idx = (torch.arange(T, device=device) - shift).clamp(min=0)
        batch_token = input_ids[batch_idx]
        batch_source = token_map[batch_token].to(torch.int64)
        batch_dead = dead_mask[batch_idx]
        batch_source = torch.where(batch_dead, torch.full_like(batch_source, dead_id), batch_source)

        # 分支 B
        col = chunk_start - 1 - lookback
        in_window = (~in_batch) & (col >= 0) & (col < lookback_depth)
        col_clamped = col.clamp(0, lookback_depth - 1)
        req_clamped = req.clamp(0, lookback_token_ids.shape[0] - 1)
        window_token = lookback_token_ids[req_clamped, col_clamped]
        window_dead = lookback_dead_mask[req_clamped, col_clamped]
        known = in_window & (window_token >= 0)
        window_source = token_map[window_token.clamp(min=0)].to(torch.int64)
        window_source = torch.where(window_dead, torch.full_like(window_source, dead_id), window_source)

        # 分支 C
        if cache is not None:
            clamped = lookback.clamp(min=0, max=max_blocks * cache_block_size - 1)
            block_row = req.clamp(max=num_table_rows - 1)
            needs_cache = (~in_batch) & (~known)
            block = block_table[block_row, clamped // cache_block_size].to(torch.int64)
            slot = (block * cache_block_size + clamped % cache_block_size).clamp(min=0, max=num_slots - 1)
            fallback = cache[slot].to(torch.int64)
            fallback = torch.where(needs_cache, fallback, pad_tensor)
        else:
            fallback = pad_tensor

        source = torch.where(in_batch, batch_source, torch.where(known, window_source, fallback))

        blocked = blocked | (lookback < 0) | (source == dead_id)
        value = torch.where(blocked, pad_tensor, source)
        rolling = rolling ^ (value[:, None] * multipliers[:, shift][None, :])


        if shift > 0:
            for h in range(H):
                c = (shift - 1) * H + h
                prime = primes[:, shift - 1, h]
                offset = offsets[:, c]
                hashed = rolling % prime[None, :] + offset[None, :]
                output[:, :, c] = hashed.to(torch.int32)

    return output


def engram_lookup_torch(
    indices,        # [T, TOTAL_HEADS] int32
    weight,         # [part_num_embeddings, DIM] float8_e4m3fn
    scales,         # [part_num_embeddings, DIM//QUANT_BLOCK] uint8
    vocab_start: int,
    vocab_end: int,
    HEAD_START: int,
    LOCAL_HEADS: int,
    TOTAL_HEADS: int,
    DIM: int,
    QUANT_BLOCK: int,
):
    T = indices.shape[0]
    out = torch.zeros((T, LOCAL_HEADS, DIM), dtype=torch.bfloat16, device=indices.device)

    # 本 rank 的 head 列
    head_idx = torch.arange(HEAD_START, HEAD_START + LOCAL_HEADS, device=indices.device)
    head_valid = head_idx < TOTAL_HEADS

    sub = indices[:, head_idx]                                  # [T, LOCAL_HEADS]
    owned = (sub >= vocab_start) & (sub < vocab_end)            # [T, LOCAL_HEADS]
    owned &= head_valid[None, :]

    # NOTE: 上界也必须钳住，否则当 sub >= vocab_end 时 index_select 会越界
    # （Triton kernel 用 mask=owned 加载所以安全；owned 掩码保证这些行最后
    # 会被清零，因此结果不变）。
    local = (sub - vocab_start).clamp(min=0, max=weight.shape[0] - 1)

    # gather weight: [T, LOCAL_HEADS, DIM]
    flat_local = local.reshape(-1)
    w = weight.index_select(0, flat_local).to(torch.float32)    # [T*LH, DIM]

    # gather scale: [T, LOCAL_HEADS, DIM//QUANT_BLOCK]
    s = scales.index_select(0, flat_local)                      # [T*LH, DIM//QUANT_BLOCK] uint8

    # ue8m0: 2^(s - 127)
    s_fp32 = (s.to(torch.int32) << 23).view(torch.float32)      # [T*LH, DIM//QUANT_BLOCK]

    # 展开到 DIM
    s_expanded = s_fp32.repeat_interleave(QUANT_BLOCK, dim=1)[:, :DIM]  # [T*LH, DIM]

    result = (w * s_expanded).to(torch.bfloat16).reshape(T, LOCAL_HEADS, DIM)
    out = torch.where(owned[:, :, None], result, torch.zeros_like(result))
    return out


@torch.no_grad()
def fused_engram_post_wkv_torch(
    hidden_states,   # [T, HC, DIM] bf16
    kv,              # [T, (HC+1)*DIM] bf16
    q_weight,        # [HC, DIM] bf16
    k_weight,        # [HC, DIM] bf16
    token_mask=None, # [T] bool 或 None
    eps=1e-20,
    clamp_value=1e-6,
):
    T, HC, DIM = hidden_states.shape
    device = hidden_states.device

    hidden = hidden_states.to(torch.float32)                        # [T, HC, DIM]
    kv = kv.to(torch.float32)

    # key: [T, HC, DIM]
    key = kv[:, :HC*DIM].reshape(T, HC, DIM)
    # value: [T, DIM]，所有 hc 共享
    value = kv[:, HC*DIM:(HC+1)*DIM]                                # [T, DIM]

    q = q_weight.to(torch.float32)                                  # [HC, DIM]
    k = k_weight.to(torch.float32)                                  # [HC, DIM]

    # RMS 归一化
    hidden_rms = torch.rsqrt(hidden.pow(2).sum(dim=-1) / DIM + eps) # [T, HC]
    key_rms = torch.rsqrt(key.pow(2).sum(dim=-1) / DIM + eps)       # [T, HC]

    # dot: [T, HC]
    dot = (hidden * q[None] * k[None] * key).sum(dim=-1)
    dot = dot * hidden_rms * key_rms * (1.0 / (DIM ** 0.5))

    # gate
    gate_input = torch.sqrt(torch.clamp(dot.abs(), min=clamp_value))
    gate_input = torch.where(dot < 0, -gate_input, gate_input)
    gate = torch.sigmoid(gate_input)                                # [T, HC]

    if token_mask is not None:
        gate = gate * token_mask[:, None].to(gate.dtype)

    # output: [T, HC, DIM]
    out = hidden + gate[:, :, None] * value[:, None, :]
    return out.to(torch.bfloat16)