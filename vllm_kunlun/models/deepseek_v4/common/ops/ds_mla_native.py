# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native XPU gather for the fp8_ds_mla paged K cache (decode path).

Wraps the pre-built ``index_select`` kernel in torch_xmlir's libxpuapi.so via a
thin custom-op shared library (``csrc/ds_mla_gather.cpp``).  The native full-block
gather is ~1.9x faster than the torch advanced-indexing gather in the decode
hot path (see DEEP_PROFILE_ATTN.md), because the Kunlun SIMD memcpy beats
torch's per-element index computation even though it copies a whole block.
"""
import os

import torch

_CACHE_DIR = "/root/.cache/torch_extensions/py310_cu118/ds_mla_gather"
_LIB = os.path.join(_CACHE_DIR, "ds_mla_gather.so")
_loaded = False

TOKEN_FP8_DIM = 448
TOKEN_BF16_DIM = 64
TOKEN_SCALE_DIM = 8
TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2  # 576
N_QUANT_BLOCKS = 7


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    if not os.path.exists(_LIB):
        raise RuntimeError(
            f"{_LIB} not found; run csrc/build_ds_mla_gather.py in the container first"
        )
    torch.ops.load_library(_LIB)
    _loaded = True


def gather_rows_scales_native(
    cache: torch.Tensor,  # [num_blocks, block_bytes] uint8
    block_size: int,
    slots: torch.Tensor,  # [n] int64 global slot ids, all >= 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather rows [n, 576] and scales [n, 7] via the native index_select kernel.

    Mirrors `gather_ds_mla_slots_torch` in cache_utils.py, but replaces the
    advanced-indexing gather with `index_select_u8(cache2d, block_ids)` (full
    block) followed by a slice. Returns uint8 tensors; dequant stays in torch.
    """
    _ensure_loaded()
    device = slots.device
    cache2d = cache.reshape(cache.shape[0], -1)
    block_ids = slots // block_size
    pos_in_block = slots % block_size

    data_off = (pos_in_block * TOKEN_DATA_SIZE).unsqueeze(1) + torch.arange(
        TOKEN_DATA_SIZE, device=device
    )
    rows_idx = torch.stack(
        [block_ids.unsqueeze(1).expand(-1, TOKEN_DATA_SIZE), data_off], dim=-1
    )
    rows = torch.ops.ds_mla_gather.gather_nd_u8(cache2d, rows_idx)  # [n, 576]

    scale_base = block_size * TOKEN_DATA_SIZE
    scale_off = (scale_base + pos_in_block * TOKEN_SCALE_DIM).unsqueeze(1) + torch.arange(
        N_QUANT_BLOCKS, device=device
    )
    sbytes_idx = torch.stack(
        [block_ids.unsqueeze(1).expand(-1, N_QUANT_BLOCKS), scale_off], dim=-1
    )
    sbytes = torch.ops.ds_mla_gather.gather_nd_u8(cache2d, sbytes_idx)  # [n, 7]

    return rows, sbytes
