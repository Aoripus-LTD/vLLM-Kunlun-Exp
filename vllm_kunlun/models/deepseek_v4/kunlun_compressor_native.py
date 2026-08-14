# SPDX-License-Identifier: Apache-2.0
"""C4(overlap, ratio=4) 压缩器的 xspeedgate compress_forward_fast 原生接入。

布局语义（独立 A/B 已验证 rel=0，见 /home/ab_compress.py）：
- ring buffer [n_req, 8, 4*head_dim] fp32，slot = [kv(2*head) | score(2*head)]，
  与分页 state_cache 的行布局完全一致，prefill→decode 切换时直接回填。
- ape_op [8, head_dim]：slot s 用 ape[s%4, (s//4)*head : (s//4+1)*head]。
- decode：plan_arg1=seq_lens(int32)，plan_arg2=None，extra_data=None。
- op 内部负责：写当前 token 状态 + 窗口满 ((pos+1)%4==0) 时输出压缩向量到 out。

仅 decode 路径使用；prefill 仍走 torch 分页路径，首个 decode 步把分页
state_cache 中最近 8 个 token 的状态回填进 ring。
"""

import torch

from vllm_kunlun.models.deepseek_v4.kunlun_cache_insert import fp32_to_e4m3_bytes

HEAD = 512
NOPE = 448
FP8_MAX = 448.0
TOKEN_STRIDE = 576
SCALE_DIM = 8
QUANT_BLOCK = 64


class NativeC4Ring:
    """每个压缩器实例一个 ring（head_dim=512, ratio=4, overlap）。"""

    def __init__(self, max_reqs: int, head_dim: int, device):
        self.head_dim = head_dim
        self.ring_size = 8
        self.slot_dim = 4 * head_dim
        self.ring = torch.zeros(
            max_reqs, self.ring_size, self.slot_dim, dtype=torch.float32, device=device
        )
        self.ready = torch.zeros(max_reqs, dtype=torch.bool, device="cpu")

    def backfill(
        self, req: int, seq_len: int, state_cache, block_table_row, block_size: int
    ):
        """把分页 state_cache 中最近 ≤8 个 token 的状态行搬进 ring[req]。"""
        start = max(0, seq_len - self.ring_size)
        positions = torch.arange(start, seq_len, device=state_cache.device)
        block_ids = (positions // block_size).clamp(0, block_table_row.shape[0] - 1)
        block_numbers = block_table_row[block_ids].long()
        rows = state_cache[block_numbers, (positions % block_size).long()]
        self.ring[req, (positions % self.ring_size).long()] = rows


def build_ape_op(ape: torch.Tensor, compress_ratio: int, head_dim: int) -> torch.Tensor:
    """ape [ratio, coff*head] -> ape_op [2*ratio, head]（slot s 用 ape[s%r, (s//r)*h:(s//r+1)*h]）。"""
    ring_size = 2 * compress_ratio
    ape_op = torch.zeros(ring_size, head_dim, dtype=torch.float32, device=ape.device)
    for s in range(ring_size):
        ape_op[s] = ape[
            s % compress_ratio,
            (s // compress_ratio) * head_dim : (s // compress_ratio + 1) * head_dim,
        ]
    return ape_op


def post_compress_store_512(
    ckv: torch.Tensor,  # [N, 512] fp32 压缩向量（边界 token）
    positions: torch.Tensor,  # [N] int64（CPU 或 XPU）
    kv_slots: torch.Tensor,  # [N] int64
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    rope_head_dim: int,
    compress_ratio: int,
) -> None:
    """RMSNorm → ue8m0 FP8 → GPT-J RoPE → 576B 写缓存（与 torch 路径同语义）。"""
    device = ckv.device
    kv2d = kv_cache.view(kv_cache.shape[0], -1)
    w_f32 = rms_norm_weight.float()
    n = ckv.shape[0]
    for i in range(n):
        kv_slot = int(kv_slots[i])
        if kv_slot < 0:
            continue
        position = int(positions[i])

        rrms = torch.rsqrt(ckv[i].square().mean() + rms_norm_eps)
        normed = ckv[i] * rrms * w_f32

        qin = normed[:NOPE].to(torch.bfloat16).float()
        g = qin.view(NOPE // QUANT_BLOCK, QUANT_BLOCK)
        amax = g.abs().amax(dim=-1).clamp(min=1e-4)
        expo = torch.ceil(torch.log2(amax / FP8_MAX))
        inv = torch.exp2(-expo)
        x = (g * inv.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
        q_bytes = fp32_to_e4m3_bytes(x.reshape(-1))
        scale_bytes = (expo + 127).clamp(0, 255).to(torch.uint8)

        compressed_pos = (position // compress_ratio) * compress_ratio
        cs = cos_sin_cache[compressed_pos]
        half = rope_head_dim // 2
        e = normed[0::2]
        o = normed[1::2]
        cos = torch.ones_like(e)
        sin = torch.zeros_like(o)
        cos[-half:] = cs[:half]
        sin[-half:] = cs[half:]
        new_e = e * cos - o * sin
        new_o = e * sin + o * cos
        rotated = torch.stack([new_e, new_o], dim=-1).flatten(0)

        kv_block = kv_slot // kv_block_size
        pos_in = kv_slot % kv_block_size
        base = pos_in * TOKEN_STRIDE
        kv2d[kv_block, base : base + NOPE] = q_bytes
        kv2d[kv_block, base + NOPE : base + TOKEN_STRIDE] = (
            rotated[-rope_head_dim:].to(torch.bfloat16).view(torch.uint8)
        )
        soff = kv_block_size * TOKEN_STRIDE + pos_in * SCALE_DIM
        kv2d[kv_block, soff : soff + NOPE // QUANT_BLOCK] = scale_bytes
        kv2d[kv_block, soff + NOPE // QUANT_BLOCK] = 0


def compress_decode_native_c4(
    ring_state: NativeC4Ring,
    ape_op: torch.Tensor,
    kv_score: torch.Tensor,  # [T, 4*head_dim] bf16/fp32（kv|score 拼接）
    positions_cpu: torch.Tensor,  # [T] cpu int64
    reqs_cpu: torch.Tensor,  # [T] cpu int32
    seq_lens: torch.Tensor,  # [T] int32 XPU（= positions+1，按 token 对齐）
    kv_slot_mapping: torch.Tensor,  # [T] int64（压缩 KV 缓存槽）
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    rope_head_dim: int,
    compress_ratio: int,
) -> None:
    """C4 decode：compress_forward_fast + 后处理（norm/rope/quant/store）。"""
    from vllm_kunlun.models.deepseek_v4.prof import prof

    num_tokens = kv_score.shape[0]
    out_native = torch.zeros(
        num_tokens, ring_state.head_dim, dtype=torch.float32, device=kv_score.device
    )
    reqs_xpu = reqs_cpu.to(kv_score.device).to(torch.int32)
    with prof("comp_forward_fast"):
        torch.ops.xspeedgate_ops.compress_forward_fast(
            ring_state.ring,
            kv_score.float(),
            out_native,
            ape_op,
            reqs_xpu,
            seq_lens.to(torch.int32),
            None,
            None,
        )
    # 边界 token 后处理
    boundary = ((positions_cpu + 1) % compress_ratio == 0).nonzero().flatten()
    if boundary.numel() == 0:
        return
    kv_slots = kv_slot_mapping[boundary.to(kv_score.device)].long().cpu()
    with prof("comp_store"):
        post_compress_store_512(
            out_native[boundary.to(kv_score.device)],
            positions_cpu[boundary],
            kv_slots,
            cos_sin_cache,
            kv_cache,
            kv_block_size,
            rms_norm_weight,
            rms_norm_eps,
            rope_head_dim,
            compress_ratio,
        )
