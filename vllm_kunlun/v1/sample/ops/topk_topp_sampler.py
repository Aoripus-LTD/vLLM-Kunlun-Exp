# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Optional

import kunlun_ops
import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)


def flashinfer_sampler_supported() -> bool:
    """FlashInfer is not supported on Kunlun XPU, always return False."""
    return False


class TopKTopPSampler(nn.Module):
    """
    Module that performs optional top-k and top-p filtering followed by
    weighted random sampling of logits.

    Implementations may update the logits tensor in-place.
    """

    def __init__(self, logprobs_mode, use_fp64_gumbel: bool = False):
        super().__init__()
        self.logprobs_mode = logprobs_mode
        self.use_fp64_gumbel = use_fp64_gumbel
        logger.info_once("Using FlashInfer for top-p & top-k sampling.")
        self.forward = self.forward_kunlun
        self.apply_top_k_top_p = apply_top_k_top_p

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        PyTorch-native implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        logits = self.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators), logits_to_return

    def forward_kunlun(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """More optimized implementation for top-k and top-p sampling."""
        if generators:
            logger.debug_once(
                "FlashInfer 0.2.3+ does not support "
                "per-request generators. Falling back to "
                "PyTorch-native implementation."
            )
            return self.forward_native(logits, generators, k, p)
        # flashinfer sampling functions expect contiguous logits.
        # In flex_attn/triton_attn fp32 inference, logits can be non-contiguous
        # because of slicing operation in logits_processor.
        #
        # NOTE(kunlun-deploy): 2026-08-16 昆仑芯部署修复 —— k/p 均为 None（即
        # top_k=0/top_p=1.0 被归一化后的默认采样）也走设备采样算子，避免
        # forward_native 的 random_sample 里 q.exponential_() 在昆仑芯上
        # CPU fallback（256 batch × 152K vocab 实测 4.89s/step，GPU 0%）。
        # flashinfer_sample 内部对 None 补 top_k=vocab / top_p=1.0，
        # 数学上等效无过滤纯随机采样。
        return flashinfer_sample(logits.contiguous(), k, p, generators), None


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        # Avoid sorting vocab for top-k only case.
        return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        if os.getenv("FAST_RANDOM_SAMPLE") == "1":
            q.uniform_()
            q = -torch.log(q)
            q = q.clamp(min=1e-12)
        else:
            q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        # NOTE(kunlun-deploy): 2026-08-16 昆仑芯部署修复 —— xspeedgate_ops
        # .inplace_exponential 在当前官方组合（xspeedgate_ops-0.0.0+torch25）
        # 中不存在（AttributeError，带 seed 请求解码即崩）。统一改用
        # uniform_ + (-log) 得到统计等价的指数噪声（FAST_RANDOM_SAMPLE 同款
        # 算法；-log(U) ~ Exp(1)），且 uniform_ 有设备 kernel（实测 0.01s）。
        for i, generator in generators.items():
            q[i].uniform_(generator=generator)
        q = -torch.log(q)
        q = q.clamp(min=1e-12)
    return probs.div_(q).argmax(dim=-1).view(-1)


def flashinfer_sample(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Sample from the logits using FlashInfer.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.

    NOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.

    NOTE: This function includes CPU-GPU synchronization, while `random_sample`
    does not. Call this function at the end of the forward pass to minimize
    the synchronization overhead.
    """
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    if k is None:
        # 无 top-k 过滤：等效 top_k=vocab（全词表）
        k = torch.full((probs.shape[0],), probs.shape[-1], dtype=torch.int32,
                       device=probs.device)
    else:
        k = k.to(torch.int32)
    if p is None:
        # 无 top-p 过滤：等效 top_p=1.0
        p = torch.ones((probs.shape[0],), dtype=torch.float32, device=probs.device)
    # 统一走 top_k_top_p 采样（None 补默认值后数学上等价于原三分支）
    next_token_ids = kunlun_ops.top_k_top_p_sampling_from_probs(
        probs, top_k=k, top_p=p, deterministic=True
    )

    return next_token_ids.view(-1)


def empty_exponential_noise_like(
    probs: torch.Tensor,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Return a tensor of exponential noise with the same shape as probs.

    Used by vllm.v1.spec_decode.llm_base_proposer for speculative decoding.
    """
    if use_fp64_gumbel:
        noise = torch.empty_like(probs, dtype=torch.float64)
    else:
        noise = torch.empty_like(probs)
    noise.exponential_()
    return noise


def sample_with_exponential_noise(
    probs: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Sample from probs using exponential noise q (Gumbel-max trick).

    Used by vllm.v1.spec_decode.llm_base_proposer for speculative decoding.
    """
    if q.dtype == probs.dtype:
        scores = probs.div_(q)
    else:
        scores = q.reciprocal_()
        scores.mul_(probs)
    return scores.argmax(dim=-1).view(-1)
