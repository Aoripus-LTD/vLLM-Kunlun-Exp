# SPDX-License-Identifier: Apache-2.0
"""MXFP4 experts for Kunlun XPU via on-the-fly dequant + torch-native MoE.

Upstream MXFP4 MoE backends are all unusable on Kunlun: CUDA variants need
NVIDIA kernels, ``XPU`` needs Intel's ``vllm_xpu_kernels``, ``EMULATION`` needs
Triton (and upstream ``dequant_mxfp4`` needs AMD quark). This module provides:

- ``dequant_mxfp4_torch``: pure-torch OCP MXFP4 (packed e2m1 + e8m0 group-32
  scales) dequantizer. bf16 output is numerically exact (every e2m1 grid
  value times a power-of-two scale is representable in bf16).
- ``KunlunEmulatedMxfp4Experts``: plugs into the upstream
  ``OCP_MXQuantizationEmulationTritonExperts`` frame but replaces the Triton
  GEMM with a torch-native per-expert loop (correctness-first; expert count
  per TP rank is small).
"""

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
    OCP_MXQuantizationEmulationTritonExperts,
)

# OCP e2m1 grid: sign(1) exp(2) man(1)
_E2M1_VALUES = (
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _e2m1_lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_E2M1_VALUES, device=device, dtype=torch.float32)


def dequant_mxfp4_torch(
    w: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize packed OCP MXFP4 weights to ``dtype``.

    Args:
        w: uint8 tensor [..., K/2]; two e2m1 values per byte, even index in the
            low nibble, odd index in the high nibble.
        scale: uint8 tensor [..., K/32]; e8m0 group scales (value = 2^(e-127)).
        dtype: output dtype (bf16 keeps the values exact).

    Returns:
        Tensor [..., K] in ``dtype``.
    """
    assert w.dtype == torch.uint8, f"expected packed uint8 weights, got {w.dtype}"
    if scale.dtype != torch.uint8:
        scale = scale.view(torch.uint8)

    k_packed = w.shape[-1]
    k = k_packed * 2

    lut = _e2m1_lut(w.device, dtype)
    packed = torch.stack([w & 0x0F, w >> 4], dim=-1)  # [..., K/2, 2]
    vals = lut[packed.reshape(*w.shape[:-1], k).long()]  # [..., K] fp32

    # e8m0 -> fp32 via IEEE bit trick: value = 2^(e-127) == (e << 23) as f32
    scale_f32 = (scale.to(torch.int32) << 23).view(torch.float32)  # [..., K/32]

    vals = vals.view(*w.shape[:-1], k // 32, 32) * scale_f32.unsqueeze(-1)
    return vals.reshape(*w.shape[:-1], k).to(dtype)


class KunlunEmulatedMxfp4Experts(OCP_MXQuantizationEmulationTritonExperts):
    """MXFP4 experts on Kunlun: on-the-fly dequant + torch-native MoE loop.

    Inherits weight/scales bookkeeping from the upstream emulation class and
    only replaces the (Triton) GEMM path, which Kunlun cannot execute.
    """

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta,
        apply_router_weight_on_input: bool,
    ):
        assert w1.dtype == torch.uint8
        assert w2.dtype == torch.uint8

        x = hidden_states
        num_local = w1.shape[0]
        topk_weights = topk_weights.to(x.dtype)

        if expert_map is not None:
            local_ids = expert_map[topk_ids]
        else:
            local_ids = topk_ids

        # SwiGLU clamp from the model config (V4: swiglu_limit=10.0):
        # gate clamps to max only, up clamps to both sides
        # (vllm SiluAndMulWithClamp semantics).
        limit = getattr(self.quant_config, "gemm1_clamp_limit", None)

        # Dequantize ONLY the activated local experts. The old path dequantized
        # the full local expert set on every forward (32 experts x 16 MB packed
        # per layer per step — the dominant decode cost), while a decode step
        # activates at most top-k (6) of them.
        uniq = torch.unique(local_ids[local_ids >= 0])
        from vllm_kunlun.models.deepseek_v4.prof import prof

        with prof("moe_dequant"):
            w1_dq = dequant_mxfp4_torch(w1[uniq], self.w1_scale_val[uniq], x.dtype)
            w2_dq = dequant_mxfp4_torch(w2[uniq], self.w2_scale_val[uniq], x.dtype)

        result = torch.zeros_like(output)
        with prof("moe_gemm"):
            for e_sel, e in enumerate(uniq.tolist()):
                mask = local_ids == e
                tok_idx, slot_idx = mask.nonzero(as_tuple=True)
                xe = x[tok_idx]
                h = xe @ w1_dq[e_sel].T
                gate, up = h.chunk(2, dim=-1)
                if limit is not None:
                    gate = gate.clamp(max=limit)
                    up = up.clamp(min=-limit, max=limit)
                ye = (F.silu(gate) * up) @ w2_dq[e_sel].T
                if not apply_router_weight_on_input:
                    ye = ye * topk_weights[tok_idx, slot_idx].unsqueeze(-1)
                result.index_add_(0, tok_idx, ye)

        output.copy_(result)
