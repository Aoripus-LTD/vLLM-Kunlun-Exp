# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 model — Kunlun XPU entry point.

Forked from upstream vllm v0.25.1 ``vllm.models.deepseek_v4`` (the XPU branch,
which is the non-CUDA reference implementation). The upstream package-level
hardware dispatch (``vllm/models/deepseek_v4/__init__.py``) falls through to the
NVIDIA branch on Kunlun (PlatformEnum.OOT), so this package is self-contained:
shared modules (attention / compressor / sparse_mla / quant_config / common)
are vendored here and the ``xpu/`` implementation files were adapted for Kunlun.
"""

from .model import DeepseekV4ForCausalLM
from .mtp import DeepSeekV4MTP
from .quant_config import DeepseekV4FP8Config

__all__ = [
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]
