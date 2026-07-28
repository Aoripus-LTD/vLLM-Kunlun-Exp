# SPDX-License-Identifier: Apache-2.0
"""FP8 block-quantized linear method for Kunlun: dequantize weights to bf16.

torch_xmlir's copy kernel cannot cast activations to float8 on Kunlun, so the
upstream dynamic FP8 activation-quant path (QuantFP8 + per_token_group_quant)
cannot run. Since every fp8 e4m3 value is exactly representable in bf16, we
dequantize FP8 block (128x128) weights to bf16 once at load time — numerically
exact — and then run the layer as an ordinary bf16 linear.
"""

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.utils import replace_parameter


def _scale_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Convert weight scales to fp32, accepting fp32 or e8m0 (uint8) storage."""
    if scale.dtype in (torch.float32, torch.float64):
        return scale.float()
    # e8m0 bytes (aliased to uint8 on torch 2.5.1): value = 2^(e-127),
    # which is exactly (e << 23) reinterpreted as fp32.
    return (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)


def dequant_fp8_block_to_bf16(
    weight: torch.Tensor, weight_scale: torch.Tensor, block: int = 128
) -> torch.Tensor:
    """Dequantize an fp8-e4m3 weight with [block, block] fp32/e8m0 scales.

    The scale layout is auto-detected: [ceil(N/B), ceil(K/B)] (row-major,
    weight orientation) or its transpose [ceil(K/B), ceil(N/B)].

    Runs on CPU if the device copy kernel rejects the dtype casts (Kunlun).
    """
    n, k = weight.shape
    scale = _scale_to_fp32(weight_scale)

    def _expand(s: torch.Tensor) -> torch.Tensor | None:
        if s.shape[0] * block >= n and s.shape[1] * block >= k:
            s = s.repeat_interleave(block, dim=0)[:n]
            return s.repeat_interleave(block, dim=1)[:, :k]
        return None

    expanded = _expand(scale)
    if expanded is None and scale.dim() == 2:
        # try the transposed layout
        expanded = _expand(scale.t())
    if expanded is None:
        raise RuntimeError(
            f"dequant_fp8_block: cannot align scale {tuple(weight_scale.shape)} "
            f"with weight {tuple(weight.shape)} (block={block})"
        )

    def _deq(dev_w, dev_s):
        return (dev_w.to(torch.bfloat16) * dev_s.to(torch.bfloat16)).to(
            torch.bfloat16
        )

    try:
        return _deq(weight, expanded)
    except RuntimeError:
        # torch_xmlir copy_kernel rejects some dtype casts on-device; the CPU
        # path is always available and this is a one-time load-time cost.
        w_cpu = weight.detach().to("cpu")
        s_cpu = expanded.detach().to("cpu")
        return _deq(w_cpu, s_cpu).to(weight.device)


class KunlunFp8BlockDequantLinearMethod(Fp8LinearMethod):
    """Dequantize FP8 block weights to bf16 at load; forward as bf16 linear."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_scale = getattr(layer, "weight_scale", None)
        if weight_scale is None:
            weight_scale = getattr(layer, "weight_scale_inv", None)
        if weight_scale is None:
            # Not a block-quantized layer (per-tensor scale or plain): keep the
            # upstream handling for those paths.
            from vllm.logger import init_logger

            init_logger(__name__).warning_once(
                "KunlunFp8BlockDequant: no block weight_scale on %s; "
                "falling back to upstream processing",
                getattr(layer, "prefix", type(layer).__name__),
            )
            return super().process_weights_after_loading(layer)
        try:
            print(
                f"[dequant-dbg] {getattr(layer, 'prefix', '?')} "
                f"weight={tuple(layer.weight.shape)} "
                f"scale={tuple(weight_scale.shape)}",
                flush=True,
            )
            w_bf16 = dequant_fp8_block_to_bf16(layer.weight.data, weight_scale.data)
        except RuntimeError as err:
            raise RuntimeError(
                f"dequant failed on layer {getattr(layer, 'prefix', '?')}: "
                f"weight={tuple(layer.weight.shape)} "
                f"scale={tuple(weight_scale.shape)}: {err}"
            ) from err
        replace_parameter(
            layer, "weight", torch.nn.Parameter(w_bf16, requires_grad=False)
        )
        # Drop quant bookkeeping so nothing downstream tries FP8 paths.
        if hasattr(layer, "weight_scale"):
            delattr(layer, "weight_scale")
        if hasattr(layer, "weight_scale_inv"):
            delattr(layer, "weight_scale_inv")
        if hasattr(layer, "input_scale"):
            delattr(layer, "input_scale")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)
