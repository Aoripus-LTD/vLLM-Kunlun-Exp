"""vllm kunlun init"""

import builtins
import importlib
import logging
import os
import sys

from vllm.logger import init_logger as init_vllm_logger

OLD_IMPORT_HOOK = builtins.__import__

# vLLM module → Kunlun replacement module
_MODULE_MAPPINGS = {
    "vllm.compilation.wrapper": "vllm_kunlun.compilation.wrapper",
    "vllm.v1.worker.utils": "vllm_kunlun.v1.worker.utils",
    "vllm.model_executor.model_loader.bitsandbytes_loader": "vllm_kunlun.models.model_loader.bitsandbytes_loader",
    "vllm.v1.sample.ops.topk_topp_sampler": "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
    "vllm.v1.sample.rejection_sampler": "vllm_kunlun.v1.sample.rejection_sampler",
    "vllm.attention.ops.merge_attn_states": "vllm_kunlun.ops.attention.merge_attn_states",
    "vllm.model_executor.models.config": "vllm_kunlun.models.config",
}


# =========================================================================
# Logger
# =========================================================================


def _configure_kunlun_logger() -> logging.Logger:
    """Reuse vLLM's handler for the vllm_kunlun logger tree."""
    vllm_logger = init_vllm_logger("vllm")
    kunlun_logger = logging.getLogger("vllm_kunlun")

    if not kunlun_logger.handlers:
        for handler in vllm_logger.handlers:
            kunlun_logger.addHandler(handler)

    kunlun_logger.setLevel(vllm_logger.getEffectiveLevel())
    kunlun_logger.propagate = False
    return kunlun_logger


# =========================================================================
# Import hook
# =========================================================================


def _apply_mtp_mamba_state_patch(module_name: str) -> None:
    """Patch MTP mamba state handling on relevant module loads.

    IMPORTANT: must not issue any import statement here - this runs inside
    the builtins.__import__ hook on every module import, and a nested import
    would re-enter the hook (infinite recursion with the xpytorch hook chain).
    The patch module is pre-loaded by _patch_mtp_mamba_state_if_loaded during
    register(), so sys.modules lookup is always sufficient.
    """
    if not (
        module_name == "vllm.v1.worker.gpu_model_runner"
        or module_name.startswith("vllm.model_executor.models.")
    ):
        return
    try:
        patch_module = sys.modules.get("vllm_kunlun.v1.sample.mtp_mamba_state_patch")
        if patch_module is None:
            return
        if module_name == "vllm.v1.worker.gpu_model_runner":
            runner_module = sys.modules.get("vllm.v1.worker.gpu_model_runner")
            if runner_module is not None:
                patch_module.patch_gpu_model_runner_module(runner_module)
        else:
            patch_module.patch_loaded_modules()
    except Exception:
        pass


def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    try:
        if module_name in _MODULE_MAPPINGS:
            if module_name in sys.modules:
                return sys.modules[module_name]
            target_module = _MODULE_MAPPINGS[module_name]
            module = importlib.import_module(target_module)
            sys.modules[module_name] = module
            sys.modules[target_module] = module
    except Exception:
        pass

    module = OLD_IMPORT_HOOK(
        module_name, globals=globals, locals=locals, fromlist=fromlist, level=level
    )
    _apply_mtp_mamba_state_patch(module_name)
    return module


# =========================================================================
# Registration steps (each step is a self-contained function)
# =========================================================================

# Tracks which registration steps have completed successfully,
# so that repeated register() calls (triggered by vLLM's multi-phase
# plugin discovery) skip already-done work instead of re-executing.
_completed_steps: set[str] = set()


def _load_native_extension(logger: logging.Logger) -> None:
    """Load _kunlun C extension to register torch.ops._C.weak_ref_tensor."""
    if "native_ext" in _completed_steps:
        return
    _completed_steps.add("native_ext")  # only attempt once
    try:
        from . import _kunlun  # noqa: F401

        logger.info("[KunlunPlugin] _kunlun native extension loaded")
    except ImportError as e:
        logger.warning("[KunlunPlugin] Failed to load _kunlun: %s", e)


def _patch_schema_utils(logger: logging.Logger) -> None:
    """Import wrapper & patch schema utilities."""
    if "schema" in _completed_steps:
        return
    from .schema import direct_register_custom_op  # noqa: F401
    from .schema import patch_annotations_for_schema  # noqa: F401

    logger.info("[KunlunPlugin] schema utils loaded and patched")
    _completed_steps.add("schema")


def _install_import_hook(logger: logging.Logger) -> None:
    """Replace builtins.__import__ to redirect vLLM modules to Kunlun."""
    if "import_hook" in _completed_steps:
        return
    builtins.__import__ = _custom_import
    logger.info("[KunlunPlugin] import_hook() ok")
    _completed_steps.add("import_hook")


def _patch_mtp_mamba_state_if_loaded(logger: logging.Logger) -> None:
    """Pre-load the MTP mamba state patch module and patch already-loaded
    gpu_model_runner / model modules. Pre-loading here (once, inside
    register()) guarantees the import-hook path (_apply_mtp_mamba_state_patch)
    can resolve it via sys.modules without issuing nested imports."""
    if "mtp_mamba_state" in _completed_steps:
        return
    try:
        from vllm_kunlun.v1.sample.mtp_mamba_state_patch import (
            patch_gpu_model_runner_module,
            patch_loaded_modules,
        )

        if "vllm.v1.worker.gpu_model_runner" in sys.modules:
            patch_gpu_model_runner_module(
                sys.modules["vllm.v1.worker.gpu_model_runner"]
            )
        for module_name in list(sys.modules):
            if module_name.startswith("vllm.model_executor.models."):
                patch_loaded_modules()
        _completed_steps.add("mtp_mamba_state")
    except Exception as e:
        logger.warning("[KunlunPlugin] mtp_mamba_state patch error: %s", e)


# =========================================================================
# Public API
# =========================================================================


def register():
    """Register the Kunlun platform.

    Called by vLLM plugin discovery before model loading.
    vLLM may invoke this multiple times during different discovery phases;
    each step tracks its own completion state via ``_completed_steps`` so
    already-succeeded work is skipped while previously-failed work (e.g.
    _patch_rotary_embedding blocked by circular import) is retried.
    """
    logger = _configure_kunlun_logger()

    first_call = "register_entered" not in _completed_steps
    if first_call:
        _completed_steps.add("register_entered")
        logger.info("[KunlunPlugin] register() pid=%s", os.getpid())

    _load_native_extension(logger)
    _patch_schema_utils(logger)  # fatal: raises on failure
    _install_import_hook(logger)  # fatal: raises on failure
    _patch_mtp_mamba_state_if_loaded(logger)

    if first_call:
        logger.info("[KunlunPlugin] register() done")
    return "vllm_kunlun.platforms.kunlun.KunlunPlatform"


def register_model():
    """Register models for training and inference."""
    from .models import register_model as _reg

    _reg()


def register_reasoning_parser():
    """Register reasoning parsers for inference."""
    from .reasoning import register_reasoning_parser as _reg_reasoning_parser

    _reg_reasoning_parser()


def register_tool_parser():
    """Register tool parsers for inference."""
    from .entrypoints.openai.tool_parsers import (
        register_tool_parser as _reg_tool_parser,
    )

    _reg_tool_parser()
