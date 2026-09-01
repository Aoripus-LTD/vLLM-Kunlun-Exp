# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baidu, Inc. and/or its affiliates
"""MiniMax H3 Turbo (LightX2V v1.0) LoRA wiring backported to vLLM-Omni 0.26.

vLLM-Omni 0.28 ships a dedicated ``load_minimax_h3_turbo_lora`` loader plus
``MiniMaxH3Pipeline._load_diffusion_lora_adapter`` / ``_validate_diffusion_lora_binding``
hooks for the published ``minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors``
adapter.  vLLM-Omni 0.26.0 has a working :class:`DiffusionLoRAManager` but
``MiniMaxH3Pipeline`` carries no LoRA wiring at all, and the 0.26 manager has no
pipeline hook (``_load_diffusion_lora_adapter`` / ``_validate_diffusion_lora_binding``),
so a pipeline-side loader would never be consulted.

When ``KUNLUN_DIFFUSION_H3_LORA=1`` is set, these patches:

* port the 0.28 ``load_minimax_h3_turbo_lora`` loader (safetensors format
  validation, diffusers->vLLM name mapping, fused gate/up packing) verbatim;
* attach ``_load_diffusion_lora_adapter`` / ``_validate_diffusion_lora_binding`` /
  ``_has_active_turbo_lora`` / ``_validate_turbo_sampling`` to the 0.26
  ``MiniMaxH3Pipeline`` and initialise ``_turbo_lora_adapter_ids``;
* give ``MiniMaxH3DiTModel`` the ``stacked_params_mapping`` declaration that
  lets the 0.26 :class:`DiffusionLoRAManager` bind separate Q/K/V adapters to
  the fused ``attn.qkv_proj`` layer (0.28 declares this on the model itself);
* teach the 0.26 :class:`DiffusionLoRAManager` the same optional pipeline
  loader/binding-validator hooks the 0.28 manager has, so the Turbo adapter is
  actually loaded and its binding completeness is checked on activation.

Semantics: the Turbo adapter is loaded as a vLLM ``LoRAModel`` and bound at
request time into vLLM-Omni's diffusion LoRA layers (``BaseLayerWithLoRA``
wrappers) -- it is **not** merged into the DiT weights.  The 0.26 diffusion
LoRA layers apply LoRA with plain torch matmul (no punica/Triton), so this
path avoids the vLLM punica kernels; Kunlun still has ``PunicaWrapperKunlun``
for the LLM-side LoRA path if it is ever needed.

Known gap: the Turbo checkpoint is a distilled 4-step sampler whose sigma
schedule comes from ``DMD2SigmaSchedule`` in 0.28.  0.26 has neither the
schedule class nor ``base_schedule`` support in ``minimax_h3_time_shift_sigmas``,
so this patch wires weights only; a real Turbo request must also port the
distilled schedule before output quality can match 0.28.

Every patch is idempotent and failure-tolerant.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import torch

_logger = logging.getLogger("vllm_kunlun.omni.lora_h3")

_KUNLUN_DIFFUSION_H3_LORA_PATCHED = False
_H3_LORA_PIPELINE_PATCH_PENDING = False
_H3_LORA_MANAGER_PATCH_PENDING = False

_TURBO_RANK = 128
_TURBO_ALPHA = 128
_TURBO_HIDDEN_SIZE = 5376
_TURBO_ATTENTION_INNER_SIZE = 7168
_TURBO_FFN_HIDDEN_SIZE = 14336
_TURBO_FILENAME = "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"
_TURBO_TARGETS = frozenset({"to_q", "to_k", "to_v", "out_proj", "fc1", "fc2"})
_TURBO_RAW_TARGET_SUFFIXES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
_TURBO_TARGET_DIMS = {
    "attn.to_q": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_k": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_v": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_out.0": (_TURBO_ATTENTION_INNER_SIZE, _TURBO_HIDDEN_SIZE),
    "ff.net.0.proj": (_TURBO_HIDDEN_SIZE, 2 * _TURBO_FFN_HIDDEN_SIZE),
    "ff.net.2": (_TURBO_FFN_HIDDEN_SIZE, _TURBO_HIDDEN_SIZE),
}
_TURBO_EXPECTED_RAW_TARGETS = frozenset(
    f"{prefix}.{block_index}.{suffix}"
    for prefix, block_count in (
        ("transformer_blocks", 50),
        ("token_refiner.refiner_blocks", 2),
    )
    for block_index in range(block_count)
    for suffix in _TURBO_RAW_TARGET_SUFFIXES
)
_TURBO_TARGET_PATTERN = (
    r"^transformer\.(?:token_refiner\.blocks|blocks)\.\d+\."
    r"(?:attn\.(?:to_q|to_k|to_v|out_proj)|mlp\.(?:fc1|fc2))$"
)

_TURBO_WEIGHTS_MAPPER = None


def _enabled() -> bool:
    return os.environ.get("KUNLUN_DIFFUSION_H3_LORA") == "1"


def _get_turbo_weights_mapper() -> Any:
    """Build (once) the diffusers->vLLM name mapper for the Turbo adapter."""
    global _TURBO_WEIGHTS_MAPPER
    if _TURBO_WEIGHTS_MAPPER is None:
        from vllm.model_executor.models.utils import WeightsMapper

        _TURBO_WEIGHTS_MAPPER = WeightsMapper(
            orig_to_new_substr={
                "token_refiner.refiner_blocks.": "token_refiner.blocks.",
                "transformer_blocks.": "blocks.",
                ".attn.to_out.0.": ".attn.out_proj.",
                ".ff.net.0.proj.": ".mlp.fc1.",
                ".ff.net.2.": ".mlp.fc2.",
                ".lora_A.default.": ".lora_A.",
                ".lora_B.default.": ".lora_B.",
            }
        )
    return _TURBO_WEIGHTS_MAPPER


# ---------------------------------------------------------------------------
# Turbo adapter loader (backported from vllm_omni 0.28
# diffusion/models/minimax_h3/lora.py)
# ---------------------------------------------------------------------------


def _select_turbo_file(artifact_path: str | Path) -> Path | None:
    path = Path(artifact_path)
    if path.is_file():
        return path if path.suffix == ".safetensors" else None
    if not path.is_dir():
        return None

    candidate = path / _TURBO_FILENAME
    return candidate if candidate.is_file() else None


def _validate_and_convert_tensors(checkpoint: Any) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    pairs: dict[str, set[str]] = {}
    raw_targets: set[str] = set()
    for name in checkpoint.keys():
        if name.endswith(_LORA_A_SUFFIX):
            raw_target = name[: -len(_LORA_A_SUFFIX)]
            side = "a"
        elif name.endswith(_LORA_B_SUFFIX):
            raw_target = name[: -len(_LORA_B_SUFFIX)]
            side = "b"
        else:
            raise ValueError(f"Unconsumed MiniMax-H3 Turbo tensor: {name!r}")
        raw_targets.add(raw_target)

        mapped_name = _get_turbo_weights_mapper().apply_list([name])[0]
        mapped_target = mapped_name.rsplit(".lora_", 1)[0]
        if mapped_target.rsplit(".", 1)[-1] not in _TURBO_TARGETS:
            raise ValueError(f"Unsupported MiniMax-H3 Turbo target: {raw_target!r}")
        target_sides = pairs.setdefault(mapped_target, set())
        if side in target_sides:
            raise ValueError(f"Duplicate MiniMax-H3 Turbo tensor for {mapped_target}.{side}")
        target_sides.add(side)

        tensor = checkpoint.get_tensor(name)
        if tensor.ndim != 2:
            raise ValueError(
                f"MiniMax-H3 Turbo LoRA tensors must be matrices, got {name}={tuple(tensor.shape)}"
            )
        suffix = next(
            (suffix for suffix in _TURBO_RAW_TARGET_SUFFIXES if raw_target.endswith(suffix)),
            None,
        )
        if suffix is None:
            raise ValueError(f"MiniMax-H3 Turbo LoRA contains unsupported target: {raw_target}")
        input_dim, output_dim = _TURBO_TARGET_DIMS[suffix]
        expected_shape = (_TURBO_RANK, input_dim) if side == "a" else (output_dim, _TURBO_RANK)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"MiniMax-H3 Turbo tensor has invalid global shape: "
                f"{name}={tuple(tensor.shape)}, expected={expected_shape}"
            )
        if side == "b" and ".ff.net.0.proj." in name:
            value, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat((gate, value), dim=0).contiguous()
        tensors[name] = tensor

    incomplete = sorted(target for target, sides in pairs.items() if sides != {"a", "b"})
    if incomplete:
        raise ValueError(f"Incomplete MiniMax-H3 Turbo LoRA pairs: {incomplete}")
    missing = sorted(_TURBO_EXPECTED_RAW_TARGETS - raw_targets)
    unexpected = sorted(raw_targets - _TURBO_EXPECTED_RAW_TARGETS)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 Turbo target set does not match the supported v1.0 artifact: "
            f"missing={len(missing)} {missing[:5]}, unexpected={len(unexpected)} {unexpected[:5]}"
        )
    return tensors


def _pack_h3_turbo_fc1(lora_model: Any) -> None:
    """Represent H3's fused gate/up projection without generic layout guesses."""

    from vllm.lora.lora_weights import PackedLoRALayerWeights

    for module_name, weights in tuple(lora_model.loras.items()):
        if not module_name.endswith(".mlp.fc1"):
            continue
        gate_b, up_b = weights.lora_b.chunk(2, dim=0)
        lora_model.loras[module_name] = PackedLoRALayerWeights(
            module_name=module_name,
            rank=weights.rank,
            lora_alphas=[weights.lora_alpha, weights.lora_alpha],
            lora_a=[weights.lora_a, weights.lora_a],
            lora_b=[gate_b.contiguous(), up_b.contiguous()],
            scaling=[weights.scaling, weights.scaling],
        )


def load_minimax_h3_turbo_lora(
    *,
    partition: str,
    lora_request: Any,
    lora_path: str | Path,
    dtype: torch.dtype,
    unsupported_offload_mode: str | None = None,
) -> tuple[Any, Any] | None:
    """Load the published LightX2V Turbo v1.0 through the legacy manager."""

    from safetensors import safe_open
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper

    lora_file = _select_turbo_file(lora_path)
    if lora_file is None:
        return None
    with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("key_format") != "minimax-h3-diffusers":
            if lora_file.name == _TURBO_FILENAME:
                raise ValueError(
                    "MiniMax-H3 Turbo v1.0 requires safetensors metadata "
                    "key_format='minimax-h3-diffusers'"
                )
            return None
        if lora_file.name != _TURBO_FILENAME:
            raise ValueError(
                f"MiniMax-H3 Turbo supports only {_TURBO_FILENAME!r}, got {lora_file.name!r}"
            )
        raw_alpha = metadata.get("alpha")
        try:
            alpha = float(raw_alpha) if raw_alpha is not None else math.nan
        except ValueError as exc:
            raise ValueError(f"MiniMax-H3 Turbo alpha must be numeric, got {raw_alpha!r}") from exc
        if alpha != _TURBO_ALPHA:
            raise ValueError(f"MiniMax-H3 Turbo v1.0 requires alpha={_TURBO_ALPHA}, got {raw_alpha!r}")
        if partition == "ref2va":
            raise ValueError("MiniMax-H3 Turbo LoRA supports FL2VA/T2VA only")
        if unsupported_offload_mode is not None:
            raise ValueError(
                f"MiniMax-H3 Turbo dynamic LoRA does not support {unsupported_offload_mode}"
            )
        tensors = _validate_and_convert_tensors(checkpoint)

    peft_helper = PEFTHelper.from_dict(
        {
            "r": _TURBO_RANK,
            "lora_alpha": _TURBO_ALPHA,
            "target_modules": _TURBO_TARGET_PATTERN,
        }
    )
    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device="cpu",
        dtype=dtype,
        weights_mapper=_get_turbo_weights_mapper(),
    )
    _pack_h3_turbo_fc1(lora_model)
    return lora_model, peft_helper


# ---------------------------------------------------------------------------
# MiniMaxH3Pipeline hooks (backported from vllm_omni 0.28
# diffusion/models/minimax_h3/pipeline_minimax_h3.py)
# ---------------------------------------------------------------------------


def _ensure_turbo_adapter_ids(self: Any) -> set[int]:
    adapter_ids = getattr(self, "_turbo_lora_adapter_ids", None)
    if adapter_ids is None:
        adapter_ids = set()
        self._turbo_lora_adapter_ids = adapter_ids
    return adapter_ids


def _pipeline_load_diffusion_lora_adapter(
    self: Any,
    *,
    lora_request: Any,
    lora_path: str | Path,
    dtype: torch.dtype,
) -> tuple[Any, Any] | None:
    # A cache eviction may be followed by a different adapter reusing the
    # same client-supplied ID. Every real load replaces the classification.
    _ensure_turbo_adapter_ids(self).discard(lora_request.lora_int_id)
    od_config = getattr(self, "od_config", None)
    offload_modes = []
    if getattr(od_config, "enable_cpu_offload", False):
        offload_modes.append("model-level CPU offload (--enable-cpu-offload)")
    if getattr(od_config, "enable_layerwise_offload", False):
        offload_modes.append("layerwise offload (--enable-layerwise-offload)")
    loaded = load_minimax_h3_turbo_lora(
        partition=self.partition,
        lora_request=lora_request,
        lora_path=lora_path,
        dtype=dtype,
        unsupported_offload_mode=" or ".join(offload_modes) or None,
    )
    if loaded is not None:
        _ensure_turbo_adapter_ids(self).add(lora_request.lora_int_id)
    return loaded


def _pipeline_validate_diffusion_lora_binding(
    self: Any,
    *,
    lora_model: Any,
    bound_lora_names: frozenset[str],
) -> None:
    if lora_model.id not in _ensure_turbo_adapter_ids(self):
        return
    missing = sorted(set(lora_model.loras) - bound_lora_names)
    if missing:
        raise ValueError(
            "MiniMax-H3 Turbo LoRA binding is incomplete: "
            f"bound={len(bound_lora_names)}/{len(lora_model.loras)}, missing={missing[:5]}"
        )


def _pipeline_has_active_turbo_lora(self: Any, sampling: Any) -> bool:
    lora_request = getattr(sampling, "lora_request", None)
    return (
        lora_request is not None
        and not math.isclose(0.0, float(getattr(sampling, "lora_scale", 0.0)))
        and lora_request.lora_int_id in _ensure_turbo_adapter_ids(self)
    )


def _pipeline_validate_turbo_sampling(self: Any, sampling: Any) -> None:
    from vllm_omni.errors import OmniClientError

    extra = sampling.extra_args or {}
    sigma_points = sampling.num_inference_steps
    if sigma_points != 5:
        raise OmniClientError(
            "MiniMax-H3 Turbo requires num_inference_steps=5 "
            "(five sigma points produce four denoiser evaluations)"
        )
    try:
        video_shift = float(extra.get("flow_shift", self.default_video_shift))
    except (TypeError, ValueError) as exc:
        raise OmniClientError("MiniMax-H3 Turbo requires flow_shift=6") from exc
    if not math.isclose(video_shift, 6.0):
        raise OmniClientError("MiniMax-H3 Turbo requires flow_shift=6")
    try:
        audio_shift = float(extra.get("audio_flow_shift", self.default_audio_shift))
    except (TypeError, ValueError) as exc:
        raise OmniClientError("MiniMax-H3 Turbo requires audio_flow_shift=3") from exc
    if not math.isclose(audio_shift, 3.0):
        raise OmniClientError("MiniMax-H3 Turbo requires audio_flow_shift=3")


def _patch_minimax_h3_pipeline() -> bool:
    """Attach Turbo LoRA hooks to the 0.26 ``MiniMaxH3Pipeline``.

    Returns ``True`` when the patch is in place (either freshly applied or
    already present).  During platform registration the ``pipeline_minimax_h3``
    module may not be importable yet; in that case the patch is marked pending
    and retried through the Kunlun post-import dispatcher.
    """
    global _H3_LORA_PIPELINE_PATCH_PENDING
    try:
        from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as _tmod
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as _pmod
    except Exception:
        _H3_LORA_PIPELINE_PATCH_PENDING = True
        return False
    cls = getattr(_pmod, "MiniMaxH3Pipeline", None)
    if cls is None:
        _H3_LORA_PIPELINE_PATCH_PENDING = True
        return False
    if getattr(cls, "_kunlun_h3_lora_patched", False):
        _H3_LORA_PIPELINE_PATCH_PENDING = False
        return True

    # 0.28 declares this on MiniMaxH3DiTModel so the diffusion LoRA manager can
    # bind separate Q/K/V adapters to the fused ``attn.qkv_proj`` layer.  0.26
    # has the same fused layer but no declaration.
    dit_cls = getattr(_tmod, "MiniMaxH3DiTModel", None)
    if dit_cls is not None and not getattr(dit_cls, "stacked_params_mapping", None):
        dit_cls.stacked_params_mapping = (
            (".attn.qkv_proj", ".attn.to_q", "q"),
            (".attn.qkv_proj", ".attn.to_k", "k"),
            (".attn.qkv_proj", ".attn.to_v", "v"),
        )

    _orig_init = cls.__init__

    def _init(self: Any, **kwargs: Any) -> None:
        _orig_init(self, **kwargs)
        _ensure_turbo_adapter_ids(self)

    cls.__init__ = _init
    cls._load_diffusion_lora_adapter = _pipeline_load_diffusion_lora_adapter
    cls._validate_diffusion_lora_binding = _pipeline_validate_diffusion_lora_binding
    cls._has_active_turbo_lora = _pipeline_has_active_turbo_lora
    cls._validate_turbo_sampling = _pipeline_validate_turbo_sampling

    # Refuse ref2va + Turbo with the same message 0.28 uses, without copying
    # 0.28's whole task-resolution rewrite.
    _orig_resolve_task = cls._resolve_task

    def _resolve_task(
        self: Any,
        requested: str | None,
        multi_modal_data: dict[str, Any],
        *,
        has_turbo_lora: bool = False,
    ) -> str:
        has_turbo_lora = has_turbo_lora or bool(getattr(self, "_kunlun_h3_turbo_requested", False))
        task = _orig_resolve_task(self, requested, multi_modal_data)
        if task == "ref2va" and has_turbo_lora:
            from vllm_omni.errors import OmniClientError

            raise OmniClientError("MiniMax-H3 Turbo LoRA supports T2VA/FL2VA requests only")
        return task

    cls._resolve_task = _resolve_task

    # 0.26 ``forward`` has no LoRA awareness.  Inject the 0.28 gating (turbo
    # sampling contract + ref2va refusal) around the original forward instead
    # of duplicating its ~170-line body.
    _orig_forward = cls.forward

    @torch.no_grad()
    def _forward(self: Any, request: Any) -> Any:
        sampling = getattr(request, "sampling_params", None)
        has_turbo = bool(sampling is not None and self._has_active_turbo_lora(sampling))
        if has_turbo:
            self._validate_turbo_sampling(sampling)
        previous = bool(getattr(self, "_kunlun_h3_turbo_requested", False))
        self._kunlun_h3_turbo_requested = has_turbo
        try:
            return _orig_forward(self, request)
        finally:
            self._kunlun_h3_turbo_requested = previous

    cls.forward = _forward

    cls._kunlun_h3_lora_patched = True
    _H3_LORA_PIPELINE_PATCH_PENDING = False
    _logger.info("MiniMax H3 Turbo LoRA: MiniMaxH3Pipeline hooks installed")
    return True


def _register_h3_lora_pipeline_import_hook() -> None:
    """Patch ``pipeline_minimax_h3`` as soon as it becomes importable."""
    target = "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3"
    try:
        from vllm_kunlun.registration.import_hooks import register_hook
    except Exception:
        return

    def _is_applied(module: Any) -> bool:
        cls = getattr(module, "MiniMaxH3Pipeline", None)
        return bool(getattr(cls, "_kunlun_h3_lora_patched", False))

    def _apply(module: Any) -> None:
        del module  # the patcher resolves the live module itself
        _patch_minimax_h3_pipeline()

    try:
        register_hook(target, _is_applied, _apply)
    except ValueError:
        # Already registered by an earlier apply call in this process.
        pass


# ---------------------------------------------------------------------------
# DiffusionLoRAManager hooks (backported from vllm_omni 0.28
# diffusion/lora/manager.py)
# ---------------------------------------------------------------------------


def _manager_load_adapter(self: Any, lora_request: Any) -> tuple[Any, Any]:
    """0.28 ``_load_adapter``: consult the pipeline's optional Turbo loader."""
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper
    from vllm.lora.utils import get_adapter_absolute_path

    if not self._expected_lora_modules:
        raise ValueError("No supported LoRA modules found in the diffusion pipeline.")

    lora_path = get_adapter_absolute_path(lora_request.lora_path)

    model_loader = getattr(self.pipeline, "_load_diffusion_lora_adapter", None)
    loaded = None
    if callable(model_loader):
        loaded = model_loader(
            lora_request=lora_request,
            lora_path=lora_path,
            dtype=self.dtype,
        )

    if loaded is None:
        peft_helper = PEFTHelper.from_local_dir(
            lora_path,
            max_position_embeddings=None,  # no need in diffusion
            tensorizer_config_dict=lora_request.tensorizer_config_dict,
        )
        lora_model = LoRAModel.from_local_checkpoint(
            lora_path,
            expected_lora_modules=self._expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=lora_request.lora_int_id,
            device="cpu",  # consistent w/ vllm's behavior
            dtype=self.dtype,
            model_vocab_size=None,
            tensorizer_config_dict=lora_request.tensorizer_config_dict,
            weights_mapper=None,
        )
    else:
        lora_model, peft_helper = loaded

    _logger.info(
        "Loaded PEFT config: r=%d, lora_alpha=%d, target_modules=%s",
        peft_helper.r,
        peft_helper.lora_alpha,
        peft_helper.target_modules,
    )
    _logger.info(
        "Loaded LoRA model: id=%d, num_modules=%d, modules=%s",
        lora_model.id,
        len(lora_model.loras),
        list(lora_model.loras.keys()),
    )
    for lora in lora_model.loras.values():
        lora.optimize()  # ref: _create_merged_loras_inplace, internal scaling
    return lora_model, peft_helper


def _manager_bind_adapter_weights(self: Any, lora_model: Any, scale: float) -> None:
    """0.28 ``_bind_adapter_weights`` with the binding-completeness validator."""
    from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights

    binding_validator = getattr(self.pipeline, "_validate_diffusion_lora_binding", None)
    lora_names_by_id = (
        {id(weights): name for name, weights in lora_model.loras.items()}
        if callable(binding_validator)
        else {}
    )
    bound_lora_names: set[str] = set()

    def _record_bound(weights: Any) -> None:
        name = lora_names_by_id.get(id(weights))
        if name is not None:
            bound_lora_names.add(name)

    # activate weights in each LoRA layer
    for full_module_name, lora_layer in self._lora_modules.items():
        lora_weights = self._get_lora_weights(lora_model, full_module_name)

        if lora_weights is None:
            n_slices = getattr(lora_layer, "n_slices", 1)
            if n_slices > 1:
                prefix, _, packed_suffix = full_module_name.rpartition(".")
                sub_suffixes = self._get_packed_sublayer_suffixes(packed_suffix, n_slices)
                if sub_suffixes is None:
                    lora_layer.reset_lora(0)
                    continue

                sub_loras: list[Any | None] = []
                any_found = False
                for sub_suffix in sub_suffixes:
                    sub_full_name = f"{prefix}.{sub_suffix}" if prefix else sub_suffix
                    sub_lora = self._get_lora_weights(lora_model, sub_full_name)
                    if sub_lora is not None:
                        any_found = True
                        # Packed layers expect plain (non-packed) subloras.
                        if isinstance(sub_lora, PackedLoRALayerWeights):
                            sub_lora = None
                    sub_loras.append(sub_lora if isinstance(sub_lora, LoRALayerWeights) else None)

                if not any_found:
                    lora_layer.reset_lora(0)
                    continue

                lora_a_list: list[torch.Tensor | None] = []
                lora_b_list: list[torch.Tensor | None] = []
                for sub_lora in sub_loras:
                    if sub_lora is None:
                        lora_a_list.append(None)
                        lora_b_list.append(None)
                        continue
                    lora_a_list.append(sub_lora.lora_a)
                    lora_b_list.append(sub_lora.lora_b * scale)

                lora_layer.set_lora(index=0, lora_a=lora_a_list, lora_b=lora_b_list)
                for sub_lora in sub_loras:
                    if sub_lora is not None:
                        _record_bound(sub_lora)
                _logger.debug(
                    "Activated packed LoRA for %s via submodules=%s (scale=%.2f)",
                    full_module_name,
                    sub_suffixes,
                    scale,
                )
            else:
                lora_layer.reset_lora(0)
            continue

        # Packed LoRA weights already provide per-slice tensors.
        if isinstance(lora_weights, PackedLoRALayerWeights):
            lora_a_list = lora_weights.lora_a
            lora_b_list = [
                None if b is None else b * scale for b in lora_weights.lora_b
            ]
            lora_layer.set_lora(index=0, lora_a=lora_a_list, lora_b=lora_b_list)
            _record_bound(lora_weights)
            _logger.debug(
                "Activated packed LoRA for %s (scale=%.2f)",
                full_module_name,
                scale,
            )
            continue

        # Fused (non-packed) weights: if the layer is multi-slice, split B.
        n_slices = getattr(lora_layer, "n_slices", 1)
        if n_slices > 1:
            output_slices = getattr(lora_layer, "output_slices", None)
            if output_slices is None:
                lora_layer.reset_lora(0)
                continue

            total = sum(output_slices)
            if lora_weights.lora_b.shape[0] != total:
                _logger.warning(
                    "Skipping LoRA for %s due to shape mismatch: "
                    "lora_b[0]=%d != sum(output_slices)=%d",
                    full_module_name,
                    lora_weights.lora_b.shape[0],
                    total,
                )
                lora_layer.reset_lora(0)
                continue

            b_splits = list(torch.split(lora_weights.lora_b, list(output_slices), dim=0))
            lora_a_list = [lora_weights.lora_a] * n_slices
            lora_b_list = [b * scale for b in b_splits]
            lora_layer.set_lora(index=0, lora_a=lora_a_list, lora_b=lora_b_list)
            _record_bound(lora_weights)
            _logger.debug(
                "Activated fused LoRA for packed layer %s (scale=%.2f)",
                full_module_name,
                scale,
            )
            continue

        scaled_lora_b = lora_weights.lora_b * scale
        lora_layer.set_lora(index=0, lora_a=lora_weights.lora_a, lora_b=scaled_lora_b)
        _record_bound(lora_weights)
        _logger.debug(
            "Activated LoRA for %s: lora_a shape=%s, lora_b shape=%s, scale=%.2f",
            full_module_name,
            lora_weights.lora_a.shape,
            lora_weights.lora_b.shape,
            scale,
        )

    if callable(binding_validator):
        binding_validator(
            lora_model=lora_model,
            bound_lora_names=frozenset(bound_lora_names),
        )


def _manager_reset_lora_layers(self: Any) -> None:
    for lora_layer in self._lora_modules.values():
        lora_layer.reset_lora(0)


def _manager_activate_adapter(self: Any, adapter_id: int, scale: float) -> None:
    """0.28 ``_activate_adapter``: reset every wrapper if binding fails."""
    if self._is_active_at_scale(adapter_id, scale):
        _logger.debug("Adapter %d already active at scale %.3f skipping", adapter_id, scale)
        return

    _logger.info("Activating adapter: id=%d", adapter_id)
    lora_model = self._registered_adapters[adapter_id]
    # Binding overwrites slot 0 incrementally. Invalidate the fast-path
    # state before the first mutation and leave every wrapper inactive if
    # any set_lora() call or model validator fails.
    self._active_adapter_id = None
    try:
        self._bind_adapter_weights(lora_model, scale)
    except Exception:
        self._reset_lora_layers()
        raise

    self._active_adapter_id = adapter_id
    self._update_adapter_scale(adapter_id, scale)


def _patch_diffusion_lora_manager() -> bool:
    """Add the 0.28 pipeline loader/binding hooks to the 0.26 LoRA manager."""
    global _H3_LORA_MANAGER_PATCH_PENDING
    try:
        from vllm_omni.diffusion.lora import manager as _mgr
    except Exception:
        _H3_LORA_MANAGER_PATCH_PENDING = True
        return False
    if getattr(_mgr, "_kunlun_h3_lora_manager_patched", False):
        _H3_LORA_MANAGER_PATCH_PENDING = False
        return True
    if not hasattr(_mgr, "DiffusionLoRAManager"):
        # The module is still being imported (circular import).  The Kunlun
        # post-import dispatcher will call us again once it finishes loading.
        _H3_LORA_MANAGER_PATCH_PENDING = True
        return False

    manager_cls = _mgr.DiffusionLoRAManager
    manager_cls._load_adapter = _manager_load_adapter
    manager_cls._bind_adapter_weights = _manager_bind_adapter_weights
    manager_cls._reset_lora_layers = _manager_reset_lora_layers
    manager_cls._activate_adapter = _manager_activate_adapter
    _mgr._kunlun_h3_lora_manager_patched = True
    _H3_LORA_MANAGER_PATCH_PENDING = False
    _logger.info("MiniMax H3 Turbo LoRA: DiffusionLoRAManager hooks installed")
    return True


def _register_h3_lora_manager_import_hook() -> None:
    """Patch ``diffusion.lora.manager`` as soon as it becomes importable."""
    target = "vllm_omni.diffusion.lora.manager"
    try:
        from vllm_kunlun.registration.import_hooks import register_hook
    except Exception:
        return

    def _is_applied(module: Any) -> bool:
        return bool(getattr(module, "_kunlun_h3_lora_manager_patched", False))

    def _apply(module: Any) -> None:
        del module  # the patcher resolves the live module itself
        _patch_diffusion_lora_manager()

    try:
        register_hook(target, _is_applied, _apply)
    except ValueError:
        # Already registered by an earlier apply call in this process.
        pass


def apply_kunlun_h3_lora_patches() -> None:
    """Apply the MiniMax H3 Turbo LoRA patches (idempotent, env-gated)."""
    global _KUNLUN_DIFFUSION_H3_LORA_PATCHED
    if _KUNLUN_DIFFUSION_H3_LORA_PATCHED:
        return
    _KUNLUN_DIFFUSION_H3_LORA_PATCHED = True
    if not _enabled():
        return

    # Register post-import hooks first (the dispatcher is already installed by
    # vllm_kunlun.register), then try direct patches for modules that are
    # already importable in this process.
    _register_h3_lora_manager_import_hook()
    _register_h3_lora_pipeline_import_hook()
    _patch_diffusion_lora_manager()
    _patch_minimax_h3_pipeline()
