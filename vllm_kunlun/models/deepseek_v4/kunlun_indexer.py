# SPDX-License-Identifier: Apache-2.0
"""Torch-native sparse indexer for DeepSeek V4 on Kunlun.

Upstream ``SparseAttnIndexer.forward_native`` only implements CUDA/ROCm/XPU
and raises NotImplementedError on Kunlun (OOT). This module reimplements the
same contract in plain torch — fp32 logits + causal topk over the fp8_ds_mla
K cache — with no fp8 casts (Kunlun's copy kernel rejects them):

- K cache rows (fp8_ds_mla, 132B/slot): first 128 bytes are e4m3 values, last
  4 bytes are the fp32 (ue8m0) group scale. Values are decoded through an
  e4m3 fp32 LUT indexed by the raw uint8 bytes.
- logits[t, j] = sum_h weights[t, h] * dot(q[t, h, :], K[j, :])
  (fp32 accumulate), where weights [T, n_head] come from weights_proj.
- Causal bound in compressed units: a token at position p may read compressed
  entries 0 .. p // compress_ratio (inclusive; its own partially-filled group
  entry is continuously overwritten by the compressor, so it is valid).
- topk = min(topk_tokens, available); when fewer valid entries exist than
  topk_tokens we return all of them (exact full attention over the cache) and
  pad the rest with -1.
"""

import torch

# e4m3 grid in fp32 (sign 1 / exp 4 / man 3, OCP e4m3fn)
_E4M3_LUT = None


def _e4m3_lut(device: torch.device) -> torch.Tensor:
    global _E4M3_LUT
    if _E4M3_LUT is None or _E4M3_LUT.device != device:
        vals = []
        for i in range(256):
            sign = -1.0 if (i >> 7) & 1 else 1.0
            exp = (i >> 3) & 0xF
            man = i & 0x7
            if exp == 0:
                v = man * (2.0**-9)
            elif exp == 15 and man == 7:
                v = float("nan")
            else:
                v = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
            vals.append(sign * v)
        _E4M3_LUT = torch.tensor(vals, dtype=torch.float32, device=device)
    return _E4M3_LUT


def dequant_fp8_ds_mla_rows(rows_u8: torch.Tensor, head_dim: int = 128) -> torch.Tensor:
    """Decode fp8_ds_mla cache rows [N, head_dim+4] (uint8) to fp32 [N, head_dim].

    Layout per row: [0:head_dim) e4m3 bytes, [head_dim:head_dim+4) fp32 scale.
    No fp8 dtype casts anywhere (Kunlun copy kernel rejects them).
    """
    vals = rows_u8[:, :head_dim]
    scale_bytes = rows_u8[:, head_dim : head_dim + 4]
    lut = _e4m3_lut(rows_u8.device)
    k = lut[vals.long()]  # [N, head_dim] fp32
    scale = scale_bytes.view(torch.float32)  # [N, 1]
    return k * scale


@torch.inference_mode()
def sparse_indexer_torch(
    k_cache_module,
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    attn_metadata,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    compress_ratio: int,
    block_size_compressed: int,
) -> torch.Tensor:
    """Torch-native replacement for SparseAttnIndexer.forward_native.

    Args:
        k_cache_module: DeepseekV4IndexerCache (owns .kv_cache [pages*?, 132]).
        q_quant: [T, n_head, head_dim] bf16 (R9 decision: bf16, unquantized).
        weights: [T, n_head] per-head logit weights.
        positions: [T] int64 token positions.
        attn_metadata: forward-context dict (or anything else -> profile no-op).
        topk_indices_buffer: [T, topk] int32/int64 buffer, mutated and returned.
        topk_tokens: number of topk entries per token.
        compress_ratio: this indexer's cache compression ratio.
        block_size_compressed: compressed entries per cache page
            (= kv_cache_spec.block_size // compress_ratio).
    """
    if topk_indices_buffer is None:
        topk_indices_buffer = torch.full(
            (q_quant.shape[0], topk_tokens),
            -1,
            dtype=torch.int32,
            device=q_quant.device,
        )
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer
    meta = attn_metadata.get(k_cache_module.prefix)
    if meta is None:
        return topk_indices_buffer

    num_tokens, n_head, head_dim = q_quant.shape
    # req id per token: the indexer-cache metadata (V3.2-style) does not carry
    # it; borrow it from any V4 attention metadata in the dict (it is a
    # per-token global field shared by all layers).
    req_id_per_token = getattr(meta, "req_id_per_token", None)
    if req_id_per_token is None:
        for v in attn_metadata.values():
            req_id_per_token = getattr(v, "req_id_per_token", None)
            if req_id_per_token is not None:
                break
    block_table = getattr(meta, "block_table", None)
    if block_table is None:
        # V3.2-style indexer metadata: block table lives in decode/prefill sub-metadata.
        dec = getattr(meta, "decode", None)
        if dec is not None and getattr(dec, "block_table", None) is not None:
            block_table = dec.block_table
        else:
            pf = getattr(meta, "prefill", None)
            chunks = getattr(pf, "chunks", None) if pf is not None else None
            if chunks:
                block_table = chunks[0].block_table
    if block_table is None:
        return topk_indices_buffer
    positions = positions.to(q_quant.device)

    out = topk_indices_buffer
    out.fill_(-1)

    kv = k_cache_module.kv_cache
    if kv.numel() == 0:
        return out
    # The indexer cache is allocated with 576B alignment per slot while the
    # meaningful payload is head_dim+4 (128 fp8 + 4B scale). Read full rows
    # and slice the payload.
    row_w = kv.shape[-1] if kv.dim() > 1 else head_dim + 4
    kv_full = kv.reshape(-1, row_w)

    lut = _e4m3_lut(q_quant.device)
    for t in range(num_tokens):
        r = int(req_id_per_token[t].item())
        p = int(positions[t].item())
        n_valid = p // compress_ratio + 1
        if n_valid <= 0:
            continue
        pages = block_table[r]  # [max_pages]
        n_pages_used = min(
            (n_valid + block_size_compressed - 1) // block_size_compressed,
            pages.shape[0],
        )
        page_ids = pages[:n_pages_used].long()
        # global slot ids for compressed entries 0..n_valid-1; clamp the page
        # table read and the final slots into the allocated cache range
        # (sentinel/garbage entries become invalid and are masked below).
        offs = torch.arange(n_valid, device=q_quant.device)
        page_slot = (offs // block_size_compressed).clamp(0, page_ids.shape[0] - 1)
        slots = (
            page_ids[page_slot] * block_size_compressed
            + offs % block_size_compressed
        ).long()
        valid_slot = (slots >= 0) & (slots < kv_full.shape[0])
        slots = slots.clamp(0, kv_full.shape[0] - 1)
        rows = kv_full[slots][:, : head_dim + 4]
        k_fp32 = lut[rows[:, :head_dim].long()] * rows[:, head_dim : head_dim + 4].view(
            torch.float32
        )
        # logits[t, j] = sum_h weights[t,h] * dot(q[t,h,:], K[j,:])
        logits_h = torch.einsum("hd,jd->hj", q_quant[t].float(), k_fp32)  # [H, J]
        logits = torch.einsum("hj,h->j", logits_h, weights[t].float())  # [J]
        logits = logits.masked_fill(~valid_slot, float("-inf"))
        k_take = min(topk_tokens, n_valid)
        if k_take == n_valid:
            take = torch.arange(n_valid, device=q_quant.device)
        else:
            take = torch.topk(logits, k_take).indices.sort().values
        out[t, :k_take] = torch.where(
            valid_slot[take],
            slots[take].to(out.dtype),
            torch.full_like(slots[take], -1, dtype=out.dtype),
        )
    return out
