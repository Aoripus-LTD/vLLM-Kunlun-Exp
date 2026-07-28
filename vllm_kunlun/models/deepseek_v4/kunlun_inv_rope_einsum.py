# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-native inverse GPT-J RoPE + grouped wo_a einsum for Kunlun.

Kunlun cannot execute Triton kernels, so this mirrors the numerics of
``vllm.v1.attention.ops.rocm_aiter_mla_sparse.rocm_inv_rope_einsum`` with
pure torch ops (correctness first).
"""

import torch


def _inverse_rope_gptj_torch(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    """bf16 inverse GPT-J (interleaved) RoPE, pure torch.

    ``o`` is ``[T, H, D]``; the first ``D - rope_head_dim`` lanes are NoPE and
    pass through (cast to bf16). The rope lanes are interleaved GPT-J style:
    for ``k`` in ``[0, rope_head_dim // 2)`` with ``a`` = lane ``2k`` and
    ``b`` = lane ``2k + 1``::

        out_even = a * cos + b * sin
        out_odd  = b * cos - a * sin

    i.e. the forward rotation with ``sin`` negated. ``cos_sin_cache`` is laid
    out as ``[P, rope_head_dim] = cos | sin``.
    """
    num_tokens, num_heads, head_dim = o.shape
    nope = head_dim - rope_head_dim
    half = rope_head_dim // 2

    out = torch.empty(
        (num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device=o.device
    )
    out[..., :nope] = o[..., :nope]

    cs = cos_sin_cache[positions.long()].to(torch.float32)  # [T, cos | sin]
    cos = cs[:, :half].unsqueeze(1)  # [T, 1, half]
    sin = cs[:, half:].unsqueeze(1)

    rope = o[..., nope:].to(torch.float32)
    a = rope[..., 0::2]  # [T, H, half]
    b = rope[..., 1::2]
    out[..., nope::2] = (a * cos + b * sin).to(torch.bfloat16)
    out[..., nope + 1 :: 2] = (b * cos - a * sin).to(torch.bfloat16)
    return out


def _get_cached_wo_a_bf16(
    wo_a: torch.nn.Module,
    n_local_groups: int,
    o_lora_rank: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Dequantize wo_a to bf16 once and cache it on the module.

    On Kunlun the FP8 linear method already dequantizes weights to bf16 at
    load time, so this is normally a plain view + cast. The ``weight_scale_inv``
    branch is kept as a fallback for modules that bypassed that path.
    """
    cached = getattr(wo_a, "_kunlun_wo_a_bf16", None)
    if cached is not None:
        return cached
    weight = wo_a.weight
    scale = getattr(wo_a, "weight_scale_inv", None)
    if scale is not None:
        from vllm_kunlun.models.deepseek_v4.kunlun_fp8_linear import (
            dequant_fp8_block_to_bf16,
        )

        weight = dequant_fp8_block_to_bf16(weight.data, scale.data)
    cached = (
        weight.view(n_local_groups, o_lora_rank, hidden_dim)
        .to(torch.bfloat16)
        .contiguous()
    )
    wo_a._kunlun_wo_a_bf16 = cached
    return cached


def kunlun_inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    """Inverse-RoPE + WO_A bmm path, pure torch (drop-in for the ROCm one)."""
    o_ref = _inverse_rope_gptj_torch(
        o, positions, rotary_emb.cos_sin_cache, rope_head_dim
    )
    o_ref = o_ref.view(o.shape[0], n_local_groups, -1)

    wo_a_weight = _get_cached_wo_a_bf16(
        wo_a, n_local_groups, o_lora_rank, o_ref.shape[-1]
    )

    return torch.einsum("tgd,grd->tgr", o_ref, wo_a_weight)
