#!/usr/bin/env python
# c_sampler_bench.py — C 阶段: 采样器微基准（真实 batch 规模）
# 用 vllm_kunlun TopKTopPSampler 测完整采样链耗时（forward_kunlun / flashinfer_sample 路径）
import os
os.environ.setdefault("VLLM_USE_V1", "1")
import time
import torch
import torch.distributed as dist
from vllm_kunlun.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

torch.manual_seed(0)
dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29888",
                        world_size=1, rank=0)
torch.cuda.set_device(0)

vocab = 152064  # Qwen3.8 词表
sampler = TopKTopPSampler(logprobs_mode="processed_logits").cuda()

def bench(batch, rounds=50):
    logits = torch.randn(batch, vocab, device="cuda", dtype=torch.float16)
    logits.requires_grad_(False)
    k = torch.tensor([20] * batch, device="cuda", dtype=torch.int32)
    p = torch.tensor([0.8] * batch, device="cuda", dtype=torch.float32)
    # warmup
    for _ in range(3):
        sampler(logits, {}, k, p)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rounds):
        sampler(logits, {}, k, p)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / rounds * 1000
    print(f"batch={batch:4d}  采样器 {dt:7.2f} ms/step  ({rounds} rounds)")

for b in (1, 8, 64, 256):
    bench(b)
