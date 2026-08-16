#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_throughput_bench.py — 吞吐压测封装（复用 vllm 官方 benchmark_throughput）

分组:
    short : 短上下文 (input-len 512 / output-len 512)  → 对齐官方 R1 5000 tok/s 基准口径
    long  : 长上下文 (input-len 32768 / output-len 256) → 业务长上下文吞吐观测

用法（TP=8 单实例，最终决策，勿改）:
    python 04_throughput_bench.py --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic --group short
    python 04_throughput_bench.py --model <目录> --group long --num-prompts 16

权威参数（2026-08-16 定稿）:
    --tensor-parallel-size 8
    --dtype float16          （勿改 bfloat16，double VRAM bug）
    --max-model-len 262144   （原生，无 YaRN）
    --quantization kl3-compressed-xline（默认。W8A8 量化模式）
    --mamba-ssm-cache-dtype float16（PR 408 必需）
    --gpu-memory-utilization 0.75
      （2026-08-16 OOM 修复：默认 0.9 会把 KV cache 预分配顶满
      86.24GiB/卡（1,331,824 tokens × 64KB），GDN SSM state 4.69GiB
      @256 并发一分配即爆；0.75 → KV ~72GiB，留 ~24GiB 给
      SSM state + activation。short/long 两组 8 卡全 OOM 实证）

输出: vllm benchmark 的标准吞吐统计（requests/s, tokens/s）
"""
import argparse
import subprocess
import sys

GROUPS = {
    "short": {"input-len": 512, "output-len": 512, "num-prompts": 256},
    "long": {"input-len": 32768, "output-len": 256, "num-prompts": 16},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--group", choices=list(GROUPS), default="short")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--mamba-ssm-cache-dtype", default="float16", help="PR 408 必需")
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--num-prompts", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=256, help="并发 batch 上限，short 组可调大")
    ap.add_argument("--quantization", default="kl3-compressed-xline",
                    help="W8A8 量化模式（默认 kl3-compressed-xline）")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.75,
                    help="KV cache 预算比例；0.9 会 OOM（见 docstring）")
    ap.add_argument("--output-json", default="bench_result.json")
    args = ap.parse_args()

    g = dict(GROUPS[args.group])
    if args.num_prompts:
        g["num-prompts"] = args.num_prompts

    # vllm 0.15.1 官方 benchmark 入口 = vllm bench throughput CLI
    # （v0.15.1 的 benchmark_throughput.py 已废弃为占位提示；vllm.benchmark /
    #   benchmark_throughput 模块在 PyPI wheel 中均不存在——2026-08-16 实测）
    candidates = [
        ["vllm", "bench", "throughput"],
        ["python3", "-m", "vllm.entrypoints.cli.main", "bench", "throughput"],
    ]
    cmd = None
    for c in candidates:
        try:
            r = subprocess.run(c + ["--help"], capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                cmd = c
                break
        except Exception:
            continue
    if cmd is None:
        sys.exit("找不到 vllm benchmark_throughput，请在 vllm-kunlun 环境内运行")

    # NOTE(kunlun-deploy): 2026-08-16 vllm 0.15.1 实测 —— bench throughput 的
    # --random-input-len/--random-output-len 有默认值（1024/128）且 random 版本
    # 优先于定长 --input-len/--output-len（warning: "The random version will be
    # preferred"），不传 random 参数会被 1024/128 覆盖。这里直接传 random
    # 版本（--random-range-ratio 默认 0.0 = 固定长度），避免被默认值顶掉。
    base = [
        *cmd,
        "--model", args.model,
        "--tensor-parallel-size", "8",  # TP=8 单实例固定
        "--dtype", args.dtype,
        "--mamba-ssm-cache-dtype", args.mamba_ssm_cache_dtype,
        "--max-model-len", str(args.max_model_len),
        "--quantization", args.quantization,
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--random-input-len", str(g["input-len"]),
        "--random-output-len", str(g["output-len"]),
        "--num-prompts", str(g["num-prompts"]),
        "--output-json", args.output_json,
    ]
    print(f"[bench] group={args.group} {g}")
    print(f"[bench] cmd: {' '.join(base)}")
    subprocess.run(base, check=True)
    print(f"[bench] 结果已写入 {args.output_json}")


if __name__ == "__main__":
    main()
