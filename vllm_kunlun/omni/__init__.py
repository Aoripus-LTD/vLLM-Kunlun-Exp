# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baidu, Inc. and/or its affiliates
"""vLLM-Omni integration for the Kunlun platform.

This package is only imported when vLLM-Omni is installed and its platform
resolver probes the ``vllm_omni.platform_plugins`` entry-point group.  It
registers :class:`vllm_kunlun.omni.platform.KunlunOmniPlatform` so the
vLLM-Omni diffusion runtime treats the Kunlun XPU (CUDA-emulated OOT
platform) as a first-class device instead of falling back to
``UnspecifiedOmniPlatform``.
"""

from typing import Optional


def register_omni_platform() -> Optional[str]:
    """Entry point for vLLM-Omni's ``vllm_omni.platform_plugins`` group.

    Returns the qualified class name of the Kunlun OmniPlatform when
    vLLM-Omni is importable, otherwise ``None`` (plugin stays inactive).
    """
    try:
        import vllm_omni  # noqa: F401
    except Exception:
        return None
    return "vllm_kunlun.omni.platform.KunlunOmniPlatform"
