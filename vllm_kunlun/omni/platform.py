# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baidu, Inc. and/or its affiliates
"""Kunlun OmniPlatform for vLLM-Omni.

The Kunlun XPU presents itself to PyTorch/vLLM as a CUDA-alike OOT device
(``device_type == "cuda"``, ``dispatch_key == "CUDA"``, ``nccl`` dist
backend), so the Omni runtime is told to use the CUDA dispatch paths.  The
only CUDA-specific pieces we override are the diffusion attention backend
selection (Kunlun has no FlashAttention/cuDNN/TensorRT kernels; the pure
torch SDPA backend is used instead) and the device/memory helpers.
"""

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.registry import (
    DiffusionAttentionBackendEnum,
)
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

from vllm_kunlun.platforms.kunlun import KunlunPlatform

logger = init_logger(__name__)

# Diffusion attention backends that require NVIDIA-specific kernels.  On
# Kunlun they are all mapped to vLLM-Omni's pure-torch SDPA backend, which
# dispatches through ``torch.nn.functional.scaled_dot_product_attention`` and
# works on the Kunlun CUDA-emulated device.
_KUNLUN_UNSUPPORTED_DIFFUSION_BACKENDS = {
    "FLASH_ATTN",
    "FLASH_ATTN_HUB",
    "FLASH_ATTN_3_HUB",
    "CUDNN_ATTN",
    "TRTLLM_ATTN",
    "FLASHINFER_ATTN",
    "SAGE_ATTN",
    "SAGE_ATTN_3",
}


class KunlunOmniPlatform(OmniPlatform, KunlunPlatform):
    """vLLM-Omni platform for the Kunlun XPU (OOT, CUDA-emulated)."""

    # Use the CUDA omni dispatch paths: the Kunlun device emulates CUDA and
    # vLLM-Omni's ``forward_cuda`` implementations (SDPA attention, RMSNorm,
    # rotary, AdaLN) all run through torch eager on Kunlun.
    _omni_enum = OmniPlatformEnum.CUDA

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_ar_worker.GPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/deploy"

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
        allow_trtllm_default: bool = False,
    ) -> str:
        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            if backend_upper in _KUNLUN_UNSUPPORTED_DIFFUSION_BACKENDS:
                logger.info(
                    "Kunlun omni: diffusion attention backend %s is not "
                    "available on Kunlun XPU; using TORCH_SDPA instead.",
                    backend_upper,
                )
                backend_upper = "TORCH_SDPA"
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.debug("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        logger.debug("Defaulting to diffusion attention backend TORCH_SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        # The Kunlun plugin runs with simple_compile_backend == "eager".
        return False

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("cuda")
        return torch.device("cuda", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.cuda.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.cuda.synchronize()

    @classmethod
    def record_device_event(cls):
        try:
            event = torch.cuda.Event()
            event.record()
            return event
        except Exception:
            logger.warning(
                "Failed to record Kunlun device event for cross-stream sync"
            )
            return None

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.cuda.mem_get_info(device)
        return free

    @classmethod
    def get_device_memory(
        cls, device: torch.device | None = None
    ) -> tuple[int, int]:
        return torch.cuda.mem_get_info(device)
