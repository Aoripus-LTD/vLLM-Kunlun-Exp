# SPDX-License-Identifier: Apache-2.0
"""ScaledMM linear kernel entries for Kunlun (PlatformEnum.OOT).

vllm 0.25.1's ``kernels/linear/__init__.py`` chooses FP8 linear kernels from
``_POSSIBLE_*_KERNELS`` dicts keyed by ``PlatformEnum``; OOT has no entries and
the lookup raises ``KeyError``. The classes here re-open the upstream Cutlass
kernels for Kunlun — their ``apply_*`` paths call ``ops.cutlass_scaled_mm``,
which vllm_kunlun already backs with ``torch.ops.xspeedgate_ops.cutlass_scaled_mm``
(see ``vllm_kunlun/ops/_custom_ops.py``), so only the platform gates need to be
relaxed.
"""

from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
)


class KunlunCutlassFp8BlockScaledMMKernel(CutlassFp8BlockScaledMMKernel):
    """FP8 block-scaled (128x128) linear kernel for Kunlun XPU.

    The upstream class gates on ``CUTLASS_BLOCK_FP8_SUPPORTED`` (CUDA sm90+).
    On Kunlun the GEMM itself is provided by the xspeedgate cutlass_scaled_mm
    op, so the platform gate is the only thing to bypass.
    """

    @classmethod
    def is_supported(cls, compute_capability=None):
        return True, None
