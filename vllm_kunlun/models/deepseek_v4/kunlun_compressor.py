# SPDX-License-Identifier: Apache-2.0
"""Torch-native fused compress→RMSNorm→RoPE→FP8-store for DeepSeek V4 (head=512).

Replaces ``compress_norm_rope_store_triton`` (sparse_attn variant) on Kunlun.

Per boundary token ((position+1) % ratio == 0):
1. Gather state-cache rows for the last (1+overlap)*ratio tokens (kv part at
   per-group head_offset, score part at +state_width), masking pos < 0.
2. softmax over the score rows (dim 0), compressed_kv = Σ w·kv (fp32).
3. RMSNorm (fp32, with weight).
4. nope 448 dims: bf16 roundtrip then ue8m0 group-64 FP8 quant
   (scale = 2^ceil(log2(amax/448)), byte = ceil+127, pad byte 0), fp8 bytes go
   to the 576B token row's first 448 bytes.
5. Full-512 GPT-J interleaved RoPE (identity outside the trailing 64 dims,
   cos/sin at (position//ratio)*ratio); trailing 64 dims stored as raw bf16.

State cache layout per token row: [kv_state(state_width) | score_state
(state_width)] fp32; the compress reads (1+overlap)*ratio rows ending at
`position`, with group g = arange(n_tok) >= ratio reading at head offset
g*head_dim inside each state's own area (coff = 1+overlap design).
"""

import torch

from vllm_kunlun.models.deepseek_v4.kunlun_cache_insert import fp32_to_e4m3_bytes

HEAD = 512
NOPE = 448
FP8_MAX = 448.0
TOKEN_STRIDE = 576
SCALE_DIM = 8
QUANT_BLOCK = 64


def compress_norm_rope_store_512_torch(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> None:
    ov = 1 if overlap else 0
    device = state_cache.device
    kv2d = kv_cache.view(kv_cache.shape[0], -1)
    head_ar = torch.arange(HEAD, device=device)
    w_f32 = rms_norm_weight.float()

    for t in range(num_actual):
        slot_id = int(slot_mapping[t].item())
        if slot_id < 0:
            continue
        position = int(positions[t].item())
        if (position + 1) % compress_ratio != 0:
            continue
        kv_slot = int(kv_slot_mapping[t].item())
        if kv_slot < 0:
            continue
        req = int(token_to_req_indices[t].item())

        n_tok = (1 + ov) * compress_ratio
        start = position - n_tok + 1
        toks = torch.arange(start, position + 1, device=device)
        valid_t = toks >= 0
        toks_c = toks.clamp(min=0)
        block_ids = (toks_c // block_size).clamp(0, block_table.shape[1] - 1)
        block_numbers = block_table[req, block_ids].long()
        offsets = toks_c % block_size
        head_off = (torch.arange(n_tok, device=device) >= compress_ratio).long() * HEAD

        rows = state_cache[block_numbers, offsets]  # [n_tok, 2*state_width] fp32
        kv_idx = head_off.unsqueeze(1) + head_ar.unsqueeze(0)  # [n_tok, HEAD]
        kv_part = torch.gather(rows, 1, kv_idx)
        sc_idx = state_width + kv_idx
        sc_part = torch.gather(rows, 1, sc_idx)

        kv_part = torch.where(valid_t.unsqueeze(1), kv_part, torch.zeros((), device=device))
        sc_part = torch.where(
            valid_t.unsqueeze(1), sc_part, torch.full((), float("-inf"), device=device)
        )
        w = torch.softmax(sc_part, dim=0)  # [n_tok, HEAD]
        ckv = (kv_part * w).sum(dim=0)  # [HEAD] fp32

        # RMSNorm (fp32)
        rrms = torch.rsqrt(ckv.square().mean() + rms_norm_eps)
        normed = ckv * rrms * w_f32

        # FP8 ue8m0 quant on nope with bf16 roundtrip first
        qin = normed[:NOPE].to(torch.bfloat16).float()
        g = qin.view(NOPE // QUANT_BLOCK, QUANT_BLOCK)
        amax = g.abs().amax(dim=-1).clamp(min=1e-4)
        expo = torch.ceil(torch.log2(amax / FP8_MAX))
        inv = torch.exp2(-expo)
        x = (g * inv.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
        q_bytes = fp32_to_e4m3_bytes(x.reshape(-1))
        scale_bytes = (expo + 127).clamp(0, 255).to(torch.uint8)

        # Full-512 GPT-J interleaved RoPE (identity outside trailing ROPE_DIM)
        compressed_pos = (position // compress_ratio) * compress_ratio
        cs = cos_sin_cache[compressed_pos]  # [ROPE_DIM]
        half = rope_head_dim // 2
        e = normed[0::2]
        o = normed[1::2]
        cos = torch.ones_like(e)
        sin = torch.zeros_like(o)
        cos[-half:] = cs[:half]
        sin[-half:] = cs[half:]
        new_e = e * cos - o * sin
        new_o = e * sin + o * cos
        rotated = torch.stack([new_e, new_o], dim=-1).flatten(0)

        # Store: fp8 448B + bf16 rope 128B + 7 scale bytes + 1 pad
        kv_block = kv_slot // kv_block_size
        pos_in = kv_slot % kv_block_size
        base = pos_in * TOKEN_STRIDE
        kv2d[kv_block, base : base + NOPE] = q_bytes
        kv2d[kv_block, base + NOPE : base + TOKEN_STRIDE] = (
            rotated[-rope_head_dim:].to(torch.bfloat16).view(torch.uint8)
        )
        soff = kv_block_size * TOKEN_STRIDE + pos_in * SCALE_DIM
        kv2d[kv_block, soff : soff + NOPE // QUANT_BLOCK] = scale_bytes
        kv2d[kv_block, soff + NOPE // QUANT_BLOCK] = 0


def compress_norm_rope_store_128_torch(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> None:
    """Indexer variant (head=128): same compress/norm/rope flow as the 512
    path, but the FP8 quant runs on the full rotated 128-vector with a single
    ue8m0 exponent, stored as 128B fp8 + 4B fp32 scale (132B slot)."""
    ov = 1 if overlap else 0
    device = state_cache.device
    HEAD128 = 128
    kv2d = kv_cache.view(kv_cache.shape[0], -1)
    head_ar = torch.arange(HEAD128, device=device)
    w_f32 = rms_norm_weight.float()

    for t in range(num_actual):
        slot_id = int(slot_mapping[t].item())
        if slot_id < 0:
            continue
        position = int(positions[t].item())
        if (position + 1) % compress_ratio != 0:
            continue
        kv_slot = int(kv_slot_mapping[t].item())
        if kv_slot < 0:
            continue
        req = int(token_to_req_indices[t].item())

        n_tok = (1 + ov) * compress_ratio
        start = position - n_tok + 1
        toks = torch.arange(start, position + 1, device=device)
        valid_t = toks >= 0
        toks_c = toks.clamp(min=0)
        block_ids = (toks_c // block_size).clamp(0, block_table.shape[1] - 1)
        block_numbers = block_table[req, block_ids].long()
        offsets = toks_c % block_size
        head_off = (torch.arange(n_tok, device=device) >= compress_ratio).long() * HEAD128

        rows = state_cache[block_numbers, offsets]
        kv_idx = head_off.unsqueeze(1) + head_ar.unsqueeze(0)
        kv_part = torch.gather(rows, 1, kv_idx)
        sc_part = torch.gather(rows, 1, state_width + kv_idx)
        kv_part = torch.where(valid_t.unsqueeze(1), kv_part, torch.zeros((), device=device))
        sc_part = torch.where(
            valid_t.unsqueeze(1), sc_part, torch.full((), float("-inf"), device=device)
        )
        w = torch.softmax(sc_part, dim=0)
        ckv = (kv_part * w).sum(dim=0)

        rrms = torch.rsqrt(ckv.square().mean() + rms_norm_eps)
        normed = ckv * rrms * w_f32

        # Full-128 GPT-J RoPE (identity outside trailing 64)
        compressed_pos = (position // compress_ratio) * compress_ratio
        cs = cos_sin_cache[compressed_pos]
        half = rope_head_dim // 2
        e = normed[0::2]
        o = normed[1::2]
        cos = torch.ones_like(e)
        sin = torch.zeros_like(o)
        cos[-half:] = cs[:half]
        sin[-half:] = cs[half:]
        new_e = e * cos - o * sin
        new_o = e * sin + o * cos
        rotated = torch.stack([new_e, new_o], dim=-1).flatten(0)

        # FP8 ue8m0 single-block quant on the rotated vector (bf16 roundtrip)
        rin = rotated.to(torch.bfloat16).float()
        absmax = rin.abs().amax().clamp(min=1e-4)
        expo = torch.ceil(torch.log2(absmax / FP8_MAX))
        inv = torch.exp2(-expo)
        x = (rin * inv).clamp(-FP8_MAX, FP8_MAX)
        q_bytes = fp32_to_e4m3_bytes(x)
        scale_val = torch.exp2(expo)

        kv_block = kv_slot // kv_block_size
        pos_in = kv_slot % kv_block_size
        base = pos_in * HEAD128
        kv2d[kv_block, base : base + HEAD128] = q_bytes
        soff = kv_block_size * HEAD128 + pos_in * 4
        kv2d[kv_block, soff : soff + 4] = scale_val.view(torch.float32).view(torch.uint8)
