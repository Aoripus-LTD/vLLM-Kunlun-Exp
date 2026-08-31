# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baidu, Inc. and/or its affiliates
"""CUDA Graph capture/replay for the MiniMax H3 denoise loop on Kunlun.

vLLM-Omni has no CUDA-graph path for diffusion and the Kunlun platform does
not support ``torch.compile``, so every denoise step pays eager Python/launch
overhead.  When ``KUNLUN_DIFFUSION_CUDAGRAPH=1`` is set, these patches:

* rewrite :meth:`MiniMaxH3DenoiseBranch.forward_kwargs` to fill static packed
  buffers instead of allocating fresh ``x``/``audio_x``/``unique_timesteps``/
  ``inverse_indices`` tensors every step (``torch.unique`` moves to the CPU,
  which also removes the GPU->CPU sync it forced);
* make ``MiniMaxH3Attention._run_packed_attention`` capture-safe by caching
  the packed ``(used, packed_total)`` limits and the boolean attention mask,
  so no ``.item()`` sync or mask allocation happens inside the graph;
* capture the packed DiT forward with ``torch.cuda.CUDAGraph`` on the first
  denoise step and replay it for every following step.

The capture region calls the module's original forward directly (bypassing
vLLM-Omni's ``_WrappedForward`` hook dispatch): model-level CPU offload is a
``SequentialOffloadHook`` whose ``pre_forward`` performs weight copies and
``torch.cuda.synchronize()``, which are illegal inside a capture.  A single
eager hooked call before capture loads the DiT weights onto the device, so
the raw forward is safe for the remainder of the loop.

Every patch is idempotent and failure-tolerant.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

_KUNLUN_DIFFUSION_CUDAGRAPH_PATCHED = False
_MAX_UNIQUE_TIMESTEPS = 4

_logger = logging.getLogger("vllm_kunlun.omni.cudagraph")


def _enabled() -> bool:
    return os.environ.get("KUNLUN_DIFFUSION_CUDAGRAPH") == "1"


# ---------------------------------------------------------------------------
# MiniMaxH3Attention capture-safety helpers
# ---------------------------------------------------------------------------


def _static_packed_limits(
    self: Any,
    cu_seqlens: torch.Tensor,
    q: torch.Tensor,
) -> tuple[int, int]:
    """Return ``(used, packed_total)`` without a GPU->CPU sync on replay."""
    key = (cu_seqlens.data_ptr(), int(cu_seqlens.numel()), int(q.shape[0]))
    cached = getattr(self, "_cg_packed_limits_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    used = int(cu_seqlens[1].item())
    packed_total = int(cu_seqlens[-1].item())
    self._cg_packed_limits_cache = (key, used, packed_total)
    return used, packed_total


def _static_attn_mask(
    self: Any,
    used: int,
    packed_total: int,
    q: torch.Tensor,
) -> torch.Tensor:
    """Return the packed ``[1, T]`` bool attention mask as a static buffer."""
    cached = getattr(self, "_cg_attn_mask_cache", None)
    if cached is not None and cached[0] == (used, packed_total) and cached[1].device == q.device:
        return cached[1]
    mask = torch.arange(packed_total, device=q.device)[None] < used
    self._cg_attn_mask_cache = ((used, packed_total), mask)
    return mask


def _patch_minimax_h3_attention_static_meta() -> None:
    """Make ``_run_packed_attention`` capture-safe (no .item()/arange per step)."""
    try:
        from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as _tmod
    except Exception:
        return
    attn_cls = getattr(_tmod, "MiniMaxH3Attention", None)
    if attn_cls is None or getattr(attn_cls, "_kunlun_cudagraph_meta_patched", False):
        return

    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

    def _run_packed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        used, packed_total = _static_packed_limits(self, cu_seqlens, q)
        attn_mask = None
        if used < packed_total:
            attn_mask = _static_attn_mask(self, used, packed_total, q)
        metadata = AttentionMetadata(
            attn_mask=attn_mask,
            extra={
                "cu_seqlens_q": cu_seqlens,
                "cu_seqlens_k": cu_seqlens,
                "max_seqlen_q": max_seqlen,
                "max_seqlen_k": max_seqlen,
            },
        )
        return self.attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            metadata,
        ).squeeze(0)

    attn_cls._run_packed_attention = _run_packed_attention
    attn_cls._kunlun_cudagraph_meta_patched = True
    _logger.info("CUDA Graph: MiniMaxH3Attention._run_packed_attention made capture-safe")


# ---------------------------------------------------------------------------
# MiniMaxH3DenoiseBranch static packed buffers
# ---------------------------------------------------------------------------


def _patch_denoise_branch_static_buffers() -> None:
    try:
        from vllm_omni.diffusion.models.minimax_h3 import denoise_loop as _dmod
    except Exception:
        return
    branch_cls = getattr(_dmod, "MiniMaxH3DenoiseBranch", None)
    if branch_cls is None or getattr(branch_cls, "_kunlun_cudagraph_buffers_patched", False):
        return

    _orig_init = branch_cls.__init__

    def _init(self, **kwargs: Any) -> None:
        _orig_init(self, **kwargs)
        device = self.x_base.device
        self._cg_unique_buf = torch.zeros(
            _MAX_UNIQUE_TIMESTEPS, dtype=torch.float32, device=device
        )
        self._cg_inverse_buf = torch.zeros(self.seq_len, dtype=torch.int64, device=device)

    def _forward_kwargs(
        self,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        t_video: float,
        t_audio: float,
        imgvid_cond_timestep: float,
        audio_ref_cond_timestep: float,
    ) -> dict[str, Any]:
        # Static buffers keep memory addresses stable across graph replays.
        x = self.x_base
        x.zero_()
        x[0].index_copy_(0, self.img_pos_dev, video_rows)
        audio_x = self.audio_x_base
        audio_x.zero_()
        audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)

        # CPU-side unique/inverse mirrors the eager GPU torch.unique(sorted=True)
        # semantics exactly, with the number of distinct values padded to a
        # fixed width so the captured graph sees a static shape.
        timesteps = torch.full((self.seq_len,), float(t_video), dtype=torch.float32)
        timesteps[self.img_pos[self.update_mask]] = t_video
        timesteps[self.img_pos[~self.update_mask]] = imgvid_cond_timestep
        timesteps[self.audio_pos[self.audio_update_mask]] = t_audio
        timesteps[self.audio_pos[~self.audio_update_mask]] = audio_ref_cond_timestep
        unique_timesteps, inverse_indices = torch.unique(
            timesteps, sorted=True, return_inverse=True
        )
        n_unique = int(unique_timesteps.shape[0])
        self._cg_unique_buf.zero_()
        self._cg_unique_buf[:n_unique].copy_(unique_timesteps)
        self._cg_inverse_buf.copy_(inverse_indices.to(torch.int64))
        return {
            **self.static_kwargs,
            "x": x,
            "audio_x": audio_x,
            "unique_timesteps": self._cg_unique_buf,
            "inverse_indices": self._cg_inverse_buf,
        }

    branch_cls.__init__ = _init
    branch_cls.forward_kwargs = _forward_kwargs
    branch_cls._kunlun_cudagraph_buffers_patched = True
    _logger.info("CUDA Graph: MiniMaxH3DenoiseBranch.forward_kwargs uses static buffers")


# ---------------------------------------------------------------------------
# Denoise loop with CUDA Graph capture/replay
# ---------------------------------------------------------------------------


class _MiniMaxH3CudaGraphRunner:
    """Per-request CUDA Graph capture/replay for the packed DiT forward."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._graph: torch.cuda.CUDAGraph | None = None
        self._out_video: torch.Tensor | None = None
        self._out_audio: torch.Tensor | None = None
        self._graph_key: tuple[Any, ...] | None = None

    @staticmethod
    def _shape_key(fk: dict[str, Any]) -> tuple[Any, ...]:
        return (
            tuple(fk["x"].shape),
            tuple(fk["audio_x"].shape),
            tuple(fk["unique_timesteps"].shape),
            tuple(fk["inverse_indices"].shape),
        )

    def _raw_forward(self, fk: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        original = getattr(self.model, "_omni_original_forward", None)
        if original is not None:
            return original(**fk)
        return self.model.forward(**fk)

    def run(self, fk: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._shape_key(fk)
        if self._graph is None or self._graph_key != key:
            # One eager hooked call loads the DiT (and offloads encoders)
            # exactly like the normal eager path, and warms lazy backend init.
            # Its outputs are the correct step-0 results: during capture the
            # ops below are only recorded, not executed.
            out_video, out_audio = self.model(**fk)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                cap_video, cap_audio = self._raw_forward(fk)
            self._graph = graph
            self._out_video = cap_video
            self._out_audio = cap_audio
            self._graph_key = key
            return out_video, out_audio

        self._graph.replay()
        return self._out_video, self._out_audio


def _patch_denoise_loop_graph() -> None:
    try:
        from vllm_omni.diffusion.models.minimax_h3 import denoise_loop as _dmod
    except Exception:
        return
    if getattr(_dmod, "_kunlun_cudagraph_loop_patched", False):
        return

    from vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral import (
        minimax_h3_euler_eta0_step,
        minimax_h3_rf_v_to_x0,
    )

    def _minimax_h3_denoise_loop(
        *,
        model: Any,
        positive: Any,
        initial_video_rows: torch.Tensor,
        initial_audio_rows: torch.Tensor,
        keyframe_cond_rows: torch.Tensor | None,
        audio_ref_rows: torch.Tensor | None = None,
        sigmas_video: list[float],
        sigmas_audio: list[float],
        device: torch.device,
        imgvid_cond_noise_aug_for_inference: float = 0.999,
        audio_cond_noise_aug_for_inference: float = 1.0,
        on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
        step_profiler: Callable[[int], AbstractContextManager] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(sigmas_video) != len(sigmas_audio):
            raise ValueError("video/audio sigma schedules must have equal length")
        if len(sigmas_video) < 2:
            raise ValueError("sigma schedules need at least 2 entries")
        n_cond = int((~positive.update_mask).sum())
        if keyframe_cond_rows is None:
            if n_cond != 0:
                raise ValueError(f"layout has {n_cond} cond rows but keyframe_cond_rows is None")
        else:
            if int(keyframe_cond_rows.shape[0]) != n_cond:
                raise ValueError(
                    f"keyframe_cond_rows {int(keyframe_cond_rows.shape[0])} != layout cond rows {n_cond}"
                )
        video_rows = initial_video_rows.to(device=device, dtype=torch.float32).clone()
        audio_rows = initial_audio_rows.to(device=device, dtype=torch.float32).clone()
        update = positive.update_mask_dev
        audio_update = positive.audio_update_mask_dev
        if int(video_rows.shape[0]) != int(positive.img_pos.shape[0]):
            raise ValueError(
                f"initial video rows {int(video_rows.shape[0])} != positive layout rows {int(positive.img_pos.shape[0])}"
            )
        if int(audio_rows.shape[0]) != int(positive.audio_pos.shape[0]):
            raise ValueError(
                f"initial audio rows {int(audio_rows.shape[0])} != positive layout rows {int(positive.audio_pos.shape[0])}"
            )
        n_audio_ref = int((~positive.audio_update_mask).sum())
        if audio_ref_rows is None:
            if n_audio_ref != 0:
                raise ValueError(f"layout has {n_audio_ref} audio ref rows but audio_ref_rows is None")
            audio_anchor = None
        else:
            if int(audio_ref_rows.shape[0]) != n_audio_ref:
                raise ValueError(
                    f"audio_ref_rows {int(audio_ref_rows.shape[0])} != layout audio ref rows {n_audio_ref}"
                )
            audio_anchor = audio_ref_rows.to(device=device, dtype=torch.float32)
        cond_anchor = (
            keyframe_cond_rows.to(device=device, dtype=torch.float32)
            if keyframe_cond_rows is not None
            else None
        )
        if cond_anchor is not None:
            video_rows[~update] = cond_anchor
        if audio_anchor is not None:
            audio_rows[~audio_update] = audio_anchor

        graph_runner = _MiniMaxH3CudaGraphRunner(model) if _enabled() else None
        step_log = os.environ.get("KUNLUN_DIFFUSION_CUDAGRAPH_STEP_LOG") == "1"

        num_steps = len(sigmas_video) - 1
        for step in range(num_steps):
            step_cm = step_profiler(step) if step_profiler is not None else nullcontext()
            t0 = time.perf_counter()
            with step_cm:
                s_v, s_v_next = sigmas_video[step], sigmas_video[step + 1]
                s_a, s_a_next = sigmas_audio[step], sigmas_audio[step + 1]
                t_v, t_a = 1.0 - s_v, 1.0 - s_a
                imgvid_cond_t = max(t_v, float(imgvid_cond_noise_aug_for_inference))
                audio_ref_cond_t = max(t_a, float(audio_cond_noise_aug_for_inference))

                fk = positive.forward_kwargs(
                    video_rows=video_rows,
                    audio_rows=audio_rows,
                    t_video=t_v,
                    t_audio=t_a,
                    imgvid_cond_timestep=imgvid_cond_t,
                    audio_ref_cond_timestep=audio_ref_cond_t,
                )
                with torch.inference_mode():
                    if graph_runner is not None:
                        v_video, v_audio = graph_runner.run(fk)
                    else:
                        v_video, v_audio = model(**fk)
                mv_video_t = v_video.float()[update]
                mv_audio_t = v_audio.float()[audio_update]

                x0_video = minimax_h3_rf_v_to_x0(
                    video_rows[update],
                    mv_video_t,
                    torch.tensor(t_v, dtype=torch.float32, device=device),
                )
                new_target = minimax_h3_euler_eta0_step(
                    video_rows[update], x0_video, sigma_curr=s_v, sigma_next=s_v_next
                )
                video_rows = video_rows.clone()
                video_rows[update] = new_target
                if cond_anchor is not None:
                    video_rows[~update] = cond_anchor

                x0_audio = minimax_h3_rf_v_to_x0(
                    audio_rows[audio_update],
                    mv_audio_t,
                    torch.tensor(t_a, dtype=torch.float32, device=device),
                )
                new_audio = minimax_h3_euler_eta0_step(
                    audio_rows[audio_update], x0_audio, sigma_curr=s_a, sigma_next=s_a_next
                )
                audio_rows = audio_rows.clone()
                audio_rows[audio_update] = new_audio
                if audio_anchor is not None:
                    audio_rows[~audio_update] = audio_anchor
                if on_step is not None:
                    on_step(step, video_rows, audio_rows)
                if step_log:
                    mode = (
                        "capture"
                        if graph_runner is not None and step == 0
                        else ("replay" if graph_runner is not None else "eager")
                    )
                    _logger.info(
                        "[cudagraph-step] step=%d mode=%s ms=%.1f",
                        step,
                        mode,
                        (time.perf_counter() - t0) * 1000.0,
                    )

        return video_rows, audio_rows

    _dmod.minimax_h3_denoise_loop = _minimax_h3_denoise_loop
    _dmod._kunlun_cudagraph_loop_patched = True
    _logger.info("CUDA Graph: minimax_h3_denoise_loop patched with capture/replay")


def apply_kunlun_cudagraph_patches() -> None:
    """Apply the MiniMax H3 CUDA Graph patches (idempotent)."""
    global _KUNLUN_DIFFUSION_CUDAGRAPH_PATCHED
    if _KUNLUN_DIFFUSION_CUDAGRAPH_PATCHED:
        return
    _patch_minimax_h3_attention_static_meta()
    _patch_denoise_branch_static_buffers()
    _patch_denoise_loop_graph()
    _KUNLUN_DIFFUSION_CUDAGRAPH_PATCHED = True
