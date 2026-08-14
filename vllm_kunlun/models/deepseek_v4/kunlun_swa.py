# SPDX-License-Identifier: Apache-2.0
"""Torch-native SWA (sliding window attention) metadata kernels for Kunlun.

Replaces the Triton kernels in upstream ``mla/sparse_swa.py``:

- ``compute_swa_indices_and_lens``: per token, causal window
  [max(pos-W+1, 0), pos] mapped through the request block table to global
  cache slot ids; out-of-window entries are -1.
- ``compute_prefill_gather_lens``: ``gather_len = query_len +
  min(prefix_len, window_size - 1)`` (single pass).
"""

import torch


def compute_swa_indices_and_lens(
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    window_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    is_valid_token: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    token_offset: int = 0,
) -> None:
    # The metadata buffer is [max_tokens, 1, window_size]; the middle dim is a
    # singleton. Squeeze it so the window axis (last dim) is window_size.
    if swa_indices.dim() == 3:
        swa_indices = swa_indices.squeeze(1)
    n = swa_indices.shape[0]
    if n == 0:
        return
    device = swa_indices.device
    token_idx = torch.arange(n, device=device) + token_offset
    n_avail = min(token_to_req_indices.shape[0], is_valid_token.shape[0])
    in_range = token_idx < n_avail
    token_idx = token_idx.clamp(0, n_avail - 1)
    valid = is_valid_token[token_idx] & in_range

    swa_lens.zero_()
    swa_indices.fill_(-1)
    if not bool(valid.any()):
        return

    reqs = token_to_req_indices[token_idx]
    qs = query_start_loc[reqs]
    qe = query_start_loc[reqs + 1]
    prefix = seq_lens[reqs] - (qe - qs)
    pos = prefix + (token_idx - qs)
    start = (pos - window_size + 1).clamp(min=0)
    end = pos + 1
    swa_len = end - start

    offs = torch.arange(swa_indices.shape[-1], device=device)
    pos_off = start.unsqueeze(1) + offs.unsqueeze(0)
    # Clamp the block_table read into range (masked-out columns are ignored
    # downstream by valid_off, mirroring the kernel's load mask).
    block_indices = (pos_off // block_size).clamp(0, block_table.shape[1] - 1)
    block_numbers = block_table[reqs.unsqueeze(1), block_indices]
    slots = block_numbers * block_size + pos_off % block_size
    valid_off = (offs.unsqueeze(0) < swa_len.unsqueeze(1)) & valid.unsqueeze(1)
    swa_indices.copy_(
        torch.where(valid_off, slots.to(swa_indices.dtype), torch.full_like(slots, -1))
    )
    swa_lens.copy_(
        torch.where(valid, swa_len.to(swa_lens.dtype), torch.zeros_like(swa_len))
    )
    import os

    if os.environ.get("DSV4_DEBUG_SWA") == "1":
        print(
            f"[SWA-COMPUTE] n={n} window={window_size} block_size={block_size} "
            f"seq_lens={seq_lens.tolist()} prefix={prefix.tolist()} pos={pos.tolist()} "
            f"swa_len={swa_len.tolist()} slots_first={slots[0][:6].tolist()} "
            f"block_table_row0={block_table[reqs[0]][:4].tolist()}",
            flush=True,
        )


class _TorchLaunchable:
    """Mimics the ``kernel[grid](...)`` Triton launch syntax for a torch fn."""

    def __init__(self, fn):
        self.fn = fn
        self._kunlun_patched = True

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            kwargs.pop("TRITON_BLOCK_SIZE", None)
            kwargs.pop("BLOCK_SIZE", None)
            return self.fn(*args, **kwargs)

        return _launch


def make_swa_indices_launchable():
    def _swa(
        swa_indices,
        swa_indices_stride,
        swa_lens,
        window_size,
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        is_valid_token,
        block_table,
        block_table_stride,
        block_size,
        token_offset=0,
        **_,
    ):
        return compute_swa_indices_and_lens(
            swa_indices,
            swa_lens,
            window_size,
            query_start_loc,
            seq_lens,
            token_to_req_indices,
            is_valid_token,
            block_table,
            block_size,
            token_offset,
        )

    return _TorchLaunchable(_swa)


def make_dspark_noncausal_launchable():
    def _swa(
        swa_indices,
        swa_indices_stride,
        swa_lens,
        window_size,
        index_width,
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        is_valid_token,
        block_table,
        block_table_stride,
        block_size,
        token_offset=0,
        **_,
    ):
        # Non-causal: block-anchored trailing window + full block.
        if swa_indices.dim() == 3:
            swa_indices = swa_indices.squeeze(1)
        n = swa_indices.shape[0]
        if n == 0:
            return
        device = swa_indices.device
        token_idx = torch.arange(n, device=device) + token_offset
        valid = is_valid_token[token_idx]
        swa_lens.zero_()
        swa_indices.fill_(-1)
        if not bool(valid.any()):
            return
        reqs = token_to_req_indices[token_idx]
        qs = query_start_loc[reqs]
        qe = query_start_loc[reqs + 1]
        sl = seq_lens[reqs]
        prefix = sl - (qe - qs)
        start = (prefix - window_size).clamp(min=0)
        end = sl
        swa_len = end - start
        offs = torch.arange(swa_indices.shape[-1], device=device)
        pos_off = start.unsqueeze(1) + offs.unsqueeze(0)
        block_indices = (pos_off // block_size).clamp(0, block_table.shape[1] - 1)
        block_numbers = block_table[reqs.unsqueeze(1), block_indices]
        slots = block_numbers * block_size + pos_off % block_size
        valid_off = (offs.unsqueeze(0) < swa_len.unsqueeze(1)) & valid.unsqueeze(1)
        swa_indices.copy_(
            torch.where(
                valid_off, slots.to(swa_indices.dtype), torch.full_like(slots, -1)
            )
        )
        swa_lens.copy_(
            torch.where(valid, swa_len.to(swa_lens.dtype), torch.zeros_like(swa_len))
        )

    return _TorchLaunchable(_swa)


def compute_prefill_gather_lens(
    pfx_gather_lens, seq_lens, query_start_loc, num_prefills, num_decodes, window_size
):
    """gather_len = query_len + min(prefix_len, window_size - 1)."""
    if num_prefills == 0:
        return
    sl = seq_lens[num_decodes : num_decodes + num_prefills]
    qs = query_start_loc[num_decodes : num_decodes + num_prefills]
    qe = query_start_loc[num_decodes + 1 : num_decodes + num_prefills + 1]
    qlen = qe - qs
    prefix = sl - qlen
    pfx_gather_lens.copy_(
        qlen + torch.minimum(prefix, torch.full_like(prefix, window_size - 1))
    )


class _TorchFn:
    def __init__(self, fn):
        self.fn = fn
        self._kunlun_patched = True

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            kwargs.pop("BLOCK_SIZE", None)
            kwargs.pop("TRITON_BLOCK_SIZE", None)
            return self.fn(*args, **kwargs)

        return _launch
