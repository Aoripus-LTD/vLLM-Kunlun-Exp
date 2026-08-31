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

import torch

_OMNI_PATCHED = False


def _patch_diffusion_output_shm() -> None:
    """Bypass the unsafe SHM/D2H output packer on Kunlun.

    The diffusion worker's async output path packs large tensors into
    shared-memory handles while doing D2H copies on a side stream.  On the
    Kunlun CUDA-emulated device this path crashes the worker natively
    (exitcode=None, no Python traceback, no KLRM/Xid).  Return the raw output
    instead, which is pickled through the regular result queue.
    """
    try:
        from vllm_omni.diffusion import ipc as _omni_ipc
    except Exception:
        return

    if getattr(_omni_ipc, "_kunlun_pack_bypass", False):
        return

    def _kunlun_pack_diffusion_output_shm(output, d2h_stream=None):
        del d2h_stream  # side-stream D2H is unsafe on Kunlun
        return output

    _omni_ipc.pack_diffusion_output_shm = _kunlun_pack_diffusion_output_shm
    _omni_ipc._kunlun_pack_bypass = True


def _patch_h3_vae_cpu_load() -> None:
    """Construct MiniMax H3 remote VAE components on CPU.

    When CPU offload is disabled (TP>1 recipes), vLLM-Omni constructs the
    diffusion pipeline inside ``with torch.device("cuda:<rank>")``.  The
    MiniMax H3 VAE remote code then builds tensors such as
    ``torch.kaiser_window`` on the XPU through torch's DeviceContext
    injection, which the Kunlun CUDA-emulated runtime does not support
    (CUDA_ERROR_NOT_SUPPORTED).  Construct the remote components on CPU;
    the adapters move them to the target device afterwards.
    """
    try:
        from vllm_omni.diffusion.models.minimax_h3 import vae as _h3_vae
    except Exception:
        return

    if getattr(_h3_vae, "_kunlun_cpu_remote_load", False):
        return

    _orig_load_remote_component = _h3_vae._load_remote_component

    def _kunlun_load_remote_component(component_path, config):
        # Inner CPU context overrides the outer cuda device context.
        with torch.device("cpu"):
            return _orig_load_remote_component(component_path, config)

    _h3_vae._load_remote_component = _kunlun_load_remote_component
    _h3_vae._kunlun_cpu_remote_load = True


def apply_kunlun_vllm_omni_patches() -> None:
    """Apply Kunlun-specific runtime patches to vLLM-Omni (idempotent).

    Each patch is independent and failure-tolerant: the platform resolver
    must never fail because a patch import is unavailable in a given process.
    """
    global _OMNI_PATCHED
    if _OMNI_PATCHED:
        return
    _patch_diffusion_output_shm()
    _patch_h3_vae_cpu_load()
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
    try:
        apply_kunlun_vllm_omni_patches()
    except Exception:
        # Platform resolution must never fail because of a runtime patch.
        pass
    return "vllm_kunlun.omni.platform.KunlunOmniPlatform"
