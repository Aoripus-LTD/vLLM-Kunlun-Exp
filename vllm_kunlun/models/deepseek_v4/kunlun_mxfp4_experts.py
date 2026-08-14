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

Decode is dominated by the fp4 -> bf16 dequant of the activated experts
(``moe_dequant``, ~14% of forward). Since the packed weights + scales are
immutable, the dequantized result for a given (layer, local-expert) is
constant across steps, so we cache it in a per-layer LRU of size
``DSV4_MOE_DQ_CACHE`` (default 6, matching top-k; 0 disables). Memory cost is
~50 MiB per expert per layer (bf16 w1+w2), so the default 6-expert cache adds
~13 GiB across 43 layers — tune down if the KV cache budget is tight.
"""

import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
    OCP_MXQuantizationEmulationTritonExperts,
)

from vllm_kunlun.models.deepseek_v4.prof import prof

# OCP e2m1 grid: sign(1) exp(2) man(1)
_E2M1_VALUES = (
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)

# Number of dequantized experts cached per MoE layer (LRU). 0 disables caching.
_MOE_DQ_CACHE_SIZE = int(os.environ.get("DSV4_MOE_DQ_CACHE", "6"))


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

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        # LRU cache: local_expert_id -> (w1_dq, w2_dq) bf16 tensors. Keyed per
        # layer instance; dequantized weights are immutable so caching is exact.
        self._dq_cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = (
            OrderedDict()
        )
        self._dq_cache_size = _MOE_DQ_CACHE_SIZE

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

        # Dequantize ONLY the activated local experts, with an LRU cache so a
        # decode stream that reuses the same experts does not re-dequantize
        # every step.
        uniq = torch.unique(local_ids[local_ids >= 0])

        with prof("moe_dequant"):
            if self._dq_cache_size <= 0 or uniq.numel() == 0:
                # Caching disabled, or no valid routing: dequantize directly.
                w1_dq = dequant_mxfp4_torch(w1[uniq], self.w1_scale_val[uniq], x.dtype)
                w2_dq = dequant_mxfp4_torch(w2[uniq], self.w2_scale_val[uniq], x.dtype)
            else:
                uniq_list = uniq.tolist()
                # Step-local map for assembly: holds every expert in ``uniq``
                # even when the LRU cache evicts some of them (prefill can
                # activate far more than ``_dq_cache_size`` experts).
                dq_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                miss_ids: list[int] = []
                for e in uniq_list:
                    if e in self._dq_cache:
                        self._dq_cache.move_to_end(e)
                        dq_map[e] = self._dq_cache[e]
                    else:
                        miss_ids.append(e)

                if miss_ids:
                    miss_t = torch.tensor(
                        miss_ids, device=w1.device, dtype=torch.long
                    )
                    w1_miss = dequant_mxfp4_torch(
                        w1[miss_t], self.w1_scale_val[miss_t], x.dtype
                    )
                    w2_miss = dequant_mxfp4_torch(
                        w2[miss_t], self.w2_scale_val[miss_t], x.dtype
                    )
                    for i, e in enumerate(miss_ids):
                        # Views suffice for this step's assembly; clone into the
                        # cache so evicted entries can free their storage.
                        dq_map[e] = (w1_miss[i], w2_miss[i])
                        self._dq_cache[e] = (
                            w1_miss[i].clone(),
                            w2_miss[i].clone(),
                        )
                    # Evict least-recently-used experts beyond the cap.
                    while len(self._dq_cache) > self._dq_cache_size:
                        self._dq_cache.popitem(last=False)

                # Assemble in ``uniq`` (sorted) order for downstream indexing.
                w1_dq = torch.stack([dq_map[e][0] for e in uniq_list])
                w2_dq = torch.stack([dq_map[e][1] for e in uniq_list])

        with prof("moe_gemm"):
            pos = torch.searchsorted(uniq, local_ids)
            flat_pos = pos.reshape(-1)
            n_rows = flat_pos.shape[0]
            hid = x.shape[-1]

            if n_rows <= 64:
                # Decode/small-batch path: bmm 批量链，避免逐 expert python 循环。
                # （大 batch 下 [T*K, 2I, H] 展开会 OOM，走分组循环。）
                x_exp = x.reshape(-1, hid)[
                    torch.arange(n_rows, device=x.device) // local_ids.shape[-1]
                ]
                if apply_router_weight_on_input:
                    x_exp = x_exp * topk_weights.reshape(-1, 1)

                w13_exp = w1_dq[flat_pos.clamp(min=0)]  # [T*K, 2I, H]
                w2_exp = w2_dq[flat_pos.clamp(min=0)]  # [T*K, H, I]
                h = torch.bmm(x_exp.unsqueeze(1), w13_exp.transpose(1, 2))
                a = torch.ops.xspeedgate_ops.clamped_swiglu(
                    h, limit if limit is not None else 1e30
                )
                y = torch.bmm(a, w2_exp.transpose(1, 2)).squeeze(1)  # [T*K, H]
                if not apply_router_weight_on_input:
                    y = y * topk_weights.reshape(-1, 1)
                # -1 填充项（无效路由）置零，不贡献输出
                y = y * (local_ids.reshape(-1, 1) >= 0).to(y.dtype)
                summed = y.view(-1, local_ids.shape[-1], hid).sum(dim=1)
                output.copy_(summed.view(output.shape))
            else:
                result = torch.zeros_like(output)
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
