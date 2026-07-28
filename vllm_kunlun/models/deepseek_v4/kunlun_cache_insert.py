# SPDX-License-Identifier: Apache-2.0
"""Torch-native qnorm+RoPE+KV fp8 insert for DeepSeek V4 on Kunlun.

Replaces the Triton kernels in ``kunlun_qnorm_rope_kv_fp8_insert.py`` and
``common/ops/cache_utils.py`` (quantize_and_insert_k_cache):

- Q: per-head RMSNorm (no weight, over 512 dims) + GPT-J interleaved RoPE on
  the trailing 64 dims, in place.
- KV: GPT-J RoPE on the trailing 64 dims.
- Cache insert (fp8_ds_mla, per-token 576B): first 448 dims quantized to
  e4m3 with ue8m0 group-64 scales (7 groups), trailing 64 dims stored raw
  bf16; per-block scales area holds 7 ue8m0 bytes + 1 pad byte per token.

e4m3 bytes are produced without any fp8 dtype cast (Kunlun copy kernel
rejects them): nearest-value lookup over the OCP e4m3 grid built in fp32.
"""

import torch

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE = ROPE_DIM // 2

TOKEN_FP8_DIM = 448
TOKEN_BF16_DIM = 64  # 64 trailing dims stored raw as bf16 (128 bytes)
TOKEN_SCALE_DIM = 8  # 7 real ue8m0 scales + 1 pad
QUANT_BLOCK = 64
TOKEN_DATA_SIZE = 576
FP8_MAX = 448.0

_E4M3_POS = None
_E4M3_MID = None


def _e4m3_pos_values(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive OCP e4m3 grid (codes 0..127) and rounding midpoints."""
    global _E4M3_POS, _E4M3_MID
    if _E4M3_POS is None or _E4M3_POS.device != device:
        vals = []
        for i in range(128):
            exp = (i >> 3) & 0xF
            man = i & 0x7
            if exp == 0:
                v = man * (2.0**-9)
            elif exp == 15 and man == 7:
                # e4m3fn code 0x7F is NaN; keep the midpoint grid monotonic by
                # using a value above the max normal (448) for searchsorted.
                v = 464.0
            else:
                v = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
            vals.append(v)
        pos = torch.tensor(vals, dtype=torch.float32, device=device)
        mid = (pos[:-1] + pos[1:]) / 2.0
        _E4M3_POS, _E4M3_MID = pos, mid
    return _E4M3_POS, _E4M3_MID


def fp32_to_e4m3_bytes(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 to nearest OCP e4m3 and return raw uint8 codes (no cast).

    e4m3fn has no inf; code 0x7F is NaN, so quantized codes are clamped to
    126 (448.0) — near-max inputs must never become the NaN byte.
    """
    _, mid = _e4m3_pos_values(x.device)
    x = x.clamp(-FP8_MAX, FP8_MAX)
    sign = (x < 0).to(torch.uint8) << 7
    ax = x.abs()
    codes = torch.searchsorted(mid, ax.contiguous()).clamp(max=126)
    return (sign | codes.to(torch.uint8)).contiguous()


def _apply_gptj_rope(x_last64: torch.Tensor, positions: torch.Tensor, cos_sin_cache):
    """GPT-J interleaved RoPE on the trailing ROPE_DIM dims (fp32 in/out).

    pairs (2i, 2i+1); new_even = e*cos - o*sin, new_odd = e*sin + o*cos.
    cos_sin_cache[pos]: [cos(32) | sin(32)].
    """
    cs = cos_sin_cache[positions]  # [T, 64] fp32
    if x_last64.dim() == 3:
        cs = cs.unsqueeze(1)  # [T, 1, 64] broadcast over heads
    cos, sin = cs[..., :HALF_ROPE], cs[..., HALF_ROPE:]
    e = x_last64[..., 0::2]
    o = x_last64[..., 1::2]
    out = torch.stack(
        [e * cos - o * sin, e * sin + o * cos], dim=-1
    ).flatten(-2)
    return out


def qnorm_rope_q_inplace(
    q: torch.Tensor, positions: torch.Tensor, cos_sin_cache, eps: float
) -> None:
    """Per-head RMSNorm (no weight) + GPT-J RoPE, in place on [T, H, 512].

    RoPE is applied only to the first ``positions.shape[0]`` rows (q may be
    padded beyond the number of real positions).
    """
    qf = q.float()
    rms = torch.rsqrt(qf.square().mean(dim=-1, keepdim=True) + eps)
    qn = qf * rms
    q[..., :NOPE_DIM] = qn[..., :NOPE_DIM].to(q.dtype)
    n = min(q.shape[0], positions.shape[0])
    q[:n, :, NOPE_DIM:] = _apply_gptj_rope(
        qn[:n, :, NOPE_DIM:], positions[:n], cos_sin_cache
    ).to(q.dtype)


def rope_kv(kv: torch.Tensor, positions: torch.Tensor, cos_sin_cache) -> torch.Tensor:
    """GPT-J RoPE on the trailing 64 dims of kv [T, 512] (returns new tensor).

    Only the first ``min(T, len(positions))`` rows are rotated (kv may be
    padded beyond the number of real positions).
    """
    n = min(kv.shape[0], positions.shape[0])
    out = kv.clone()
    out[:n, NOPE_DIM:] = _apply_gptj_rope(
        kv[:n, NOPE_DIM:].float(), positions[:n], cos_sin_cache
    ).to(kv.dtype)
    return out


def quantize_and_insert_k_cache_torch(
    k: torch.Tensor,  # [num_tokens, 512] bf16 (RoPE applied)
    k_cache: torch.Tensor,  # [num_blocks, block_bytes] uint8
    slot_mapping: torch.Tensor,  # [num_tokens] int64
    block_size: int = 64,
) -> None:
    """FP8 (448 dims, ue8m0 group-64) + raw bf16 (64 dims) paged insert."""
    num_tokens = slot_mapping.shape[0]
    block_stride = k_cache.stride(0)
    kf = k[:num_tokens].float()

    # --- fp8 part with ue8m0 group scales ---
    g = kf[:, :TOKEN_FP8_DIM].view(num_tokens, TOKEN_FP8_DIM // QUANT_BLOCK, QUANT_BLOCK)
    amax = g.abs().amax(dim=-1).clamp(min=1e-12)  # [T, 7]
    scale_log2 = torch.ceil(torch.log2(amax / FP8_MAX))
    scale = torch.exp2(scale_log2)  # [T, 7]
    q = (g / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    q_bytes = fp32_to_e4m3_bytes(q.reshape(num_tokens, TOKEN_FP8_DIM))
    scale_bytes = (scale_log2.to(torch.uint8) + 127).to(torch.uint8)  # ue8m0

    # --- raw bf16 part ---
    bf16_bytes = k[:num_tokens, TOKEN_FP8_DIM:].contiguous().view(torch.uint8).view(
        num_tokens, TOKEN_BF16_DIM * 2
    )

    # --- write into pages (2-D indexing; no flat view so non-contiguous
    # cache layouts are fine) ---
    slots = slot_mapping[:num_tokens].long()
    block_ids = slots // block_size
    token_in_block = slots % block_size
    token_off = token_in_block * TOKEN_DATA_SIZE
    scale_off = block_size * TOKEN_DATA_SIZE + token_in_block * TOKEN_SCALE_DIM

    for t in range(num_tokens):
        if slots[t] < 0:
            continue
        b = int(block_ids[t].item())
        o = int(token_off[t].item())
        k_cache[b, o : o + TOKEN_FP8_DIM] = q_bytes[t]
        k_cache[b, o + TOKEN_FP8_DIM : o + TOKEN_DATA_SIZE] = bf16_bytes[t]
        so = int(scale_off[t].item())
        k_cache[b, so : so + TOKEN_SCALE_DIM - 1] = scale_bytes[t]
        k_cache[b, so + TOKEN_SCALE_DIM - 1] = 0
