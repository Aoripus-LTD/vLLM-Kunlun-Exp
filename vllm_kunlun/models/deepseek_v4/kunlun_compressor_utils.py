# SPDX-License-Identifier: Apache-2.0
"""Torch-native compressed slot mapping for Kunlun (replaces the Triton kernel).

Same math as upstream ``get_compressed_slot_mapping`` (mla/compressor_utils):
for each token in the query, its absolute position is
``start_pos + i`` with ``start_pos = seq_len - query_len``; a slot is written
only when the position completes a compression group
(``(pos + 1) % compress_ratio == 0``), pointing at
``block_table[req, cpos // block_size] * block_size + cpos % block_size``;
everything else stays PAD (-1).
"""

import torch


def get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    for b in range(block_table.shape[0]):
        qs = int(query_start_loc[b].item())
        qe = int(query_start_loc[b + 1].item())
        qlen = qe - qs
        if qlen <= 0:
            continue
        start_pos = int(seq_lens[b].item()) - qlen
        i = torch.arange(qlen, device=block_table.device)
        pos = start_pos + i
        valid = (pos + 1) % compress_ratio == 0
        cpos = pos // compress_ratio
        block_ids = (cpos // block_size).clamp(min=0)
        block_numbers = block_table[b, block_ids]
        slot_ids = (block_numbers * block_size + cpos % block_size).to(
            slot_mapping.dtype
        )
        slot_mapping[qs:qe] = torch.where(valid, slot_ids, torch.full_like(slot_ids, -1))
    return slot_mapping


def build_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    query_slice: slice | None = None,
    skip_kv_gather: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
):
    """Torch-native prefill chunk metadata (DCP=1 only; Kunlun runs TP, not DCP).

    Same index math as upstream ``_build_prefill_chunk_metadata_kernel``:
    per-token context lengths are ``(start_pos + 1 + i) // compress_ratio``,
    ``token_to_seq`` maps compressed entries back to request ids.
    """
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerPrefillChunkMetadata,
    )

    assert dcp_world_size == 1, "torch fallback only supports dcp_world_size=1"
    total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item()
    if total_seq_lens == 0:
        return None

    num_reqs = end_idx - start_idx
    device = block_table.device
    token_to_seq = torch.empty(total_seq_lens, dtype=torch.int32, device=device)

    cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    cu_seq_lens[:1] = 0
    torch.cumsum(compressed_seq_lens[start_idx:end_idx], dim=0, out=cu_seq_lens[1:])
    local_cu_seq_lens = cu_seq_lens
    local_total_seq_lens = total_seq_lens
    max_local_total_seq_lens = total_seq_lens

    qsl = query_start_loc[start_idx : end_idx + 1] - query_start_loc[start_idx]
    total_query_len = int(
        (query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item()
    )
    qs_start = query_slice.start if query_slice is not None else 0
    qs_stop = query_slice.stop if query_slice is not None else total_query_len
    output_query_len = qs_stop - qs_start

    cu_seq_len_ks = torch.empty(output_query_len, dtype=torch.int32, device=device)
    cu_seq_len_ke = torch.empty(output_query_len, dtype=torch.int32, device=device)

    for i, b in enumerate(range(start_idx, end_idx)):
        query_start = int(qsl[i].item())
        query_end = int(qsl[i + 1].item())
        query_len = query_end - query_start
        row_start = int(cu_seq_lens[i].item())
        uncompressed_seq_len = int(uncompressed_seq_lens[b].item())
        start_pos = uncompressed_seq_len - query_len
        if query_len > 0:
            offs = torch.arange(query_len, device=device)
            abs_pos = query_start + offs
            mask = (abs_pos >= qs_start) & (abs_pos < qs_stop)
            out_pos = abs_pos[mask] - qs_start
            cu_seq_len_ks[out_pos] = row_start
            cu_seq_len_ke[out_pos] = (
                row_start + (start_pos + 1 + offs[mask]) // compress_ratio
            )
        seq_start = int(cu_seq_lens[i].item())
        seq_end = int(cu_seq_lens[i + 1].item())
        if seq_end > seq_start:
            token_to_seq[seq_start:seq_end] = i

    token_start = query_start_loc_cpu[start_idx].item()
    if query_slice is not None:
        token_end = token_start + qs_stop
        token_start = token_start + qs_start
        skip_kv_gather = skip_kv_gather or qs_start > 0
    else:
        token_end = query_start_loc_cpu[end_idx].item()

    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seq_len_ks,
        cu_seqlen_ke=cu_seq_len_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather,
        local_cu_seq_lens=local_cu_seq_lens,
        local_total_seq_lens=local_total_seq_lens,
        max_local_total_seq_lens=max_local_total_seq_lens,
    )


def build_c128a_topk_metadata(
    positions: torch.Tensor,
    compress_ratio: int,
    num_decode_tokens: int,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    global_decode_buffer: torch.Tensor,
    decode_lens_buffer: torch.Tensor,
    prefill_buffer: torch.Tensor,
    max_compressed_tokens: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch-native C128A topk metadata (replaces the Triton kernel).

    Decode: compressed index -> block_table -> global slot ids (+ valid counts,
    zeroed when slot_mapping < 0). Prefill: local indices [0..n-1, -1...].
    """
    num_tokens = positions.shape[0]
    num_prefill_tokens = num_tokens - num_decode_tokens

    global_decode = global_decode_buffer[:num_decode_tokens]
    decode_lens = decode_lens_buffer[:num_decode_tokens]
    prefill_local = prefill_buffer[:num_prefill_tokens]

    if num_tokens == 0:
        return global_decode, decode_lens, prefill_local

    device = positions.device
    ncomp = ((positions + 1) // compress_ratio).clamp(max=max_compressed_tokens)
    offs = torch.arange(max_compressed_tokens, device=device)

    if num_decode_tokens > 0:
        nd = ncomp[:num_decode_tokens]
        valid = offs.unsqueeze(0) < nd.unsqueeze(1)  # [D, M]
        block_indices = offs // block_size
        reqs = token_to_req_indices[:num_decode_tokens].unsqueeze(1)
        block_numbers = block_table[reqs, block_indices.unsqueeze(0)]
        slots = block_numbers * block_size + offs % block_size
        global_decode_buffer[:num_decode_tokens] = torch.where(
            valid, slots.to(global_decode_buffer.dtype), torch.full_like(slots, -1)
        )
        tok_valid = slot_mapping[:num_decode_tokens] >= 0
        decode_lens_buffer[:num_decode_tokens] = torch.where(
            tok_valid, nd.to(decode_lens_buffer.dtype), torch.zeros_like(nd)
        )

    if num_prefill_tokens > 0:
        np_ = ncomp[num_decode_tokens:]
        valid_p = offs.unsqueeze(0) < np_.unsqueeze(1)
        prefill_buffer[:num_prefill_tokens] = torch.where(
            valid_p,
            offs.unsqueeze(0).to(prefill_buffer.dtype),
            torch.full_like(valid_p, -1, dtype=prefill_buffer.dtype),
        )

    return global_decode, decode_lens, prefill_local
