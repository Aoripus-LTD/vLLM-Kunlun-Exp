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

_OMNI_PATCHED = False


def apply_kunlun_vllm_omni_patches() -> None:
    """Apply Kunlun-specific runtime patches to vLLM-Omni (idempotent).

    The diffusion worker's async output path packs large tensors into
    shared-memory handles while doing D2H copies on a side stream.  On the
    Kunlun CUDA-emulated device this path crashes the worker natively
    (exitcode=None, no Python traceback, no KLRM/Xid).  Until a Kunlun-safe
    SHM packer is implemented, bypass the packing entirely and return the
    raw output, which is pickled through the regular result queue.
    """
    global _OMNI_PATCHED
    if _OMNI_PATCHED:
        return
    try:
        from vllm_omni.diffusion import ipc as _omni_ipc
    except Exception:
        return

    if getattr(_omni_ipc, "_kunlun_pack_bypass", False):
        _OMNI_PATCHED = True
        return

    def _kunlun_pack_diffusion_output_shm(output, d2h_stream=None):
        del d2h_stream  # side-stream D2H is unsafe on Kunlun
        return output

    _omni_ipc.pack_diffusion_output_shm = _kunlun_pack_diffusion_output_shm
    _omni_ipc._kunlun_pack_bypass = True
    _OMNI_PATCHED = True


def register_omni_platform() -> Optional[str]:
    """Entry point for vLLM-Omni's ``vllm_omni.platform_plugins`` group.

    Returns the qualified class name of the Kunlun OmniPlatform when
    vLLM-Omni is importable, otherwise ``None`` (plugin stays inactive).
    """
    try:
        import vllm_omni  # noqa: F401
    except Exception:
        return None
    apply_kunlun_vllm_omni_patches()
    return "vllm_kunlun.omni.platform.KunlunOmniPlatform"
