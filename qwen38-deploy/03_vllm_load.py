#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_vllm_load.py — vllm-kunlun 加载验证 + 短/长上下文生成测试（TP=8 单实例）

前提: 服务器已装好 vllm-kunlun (0.15.1.dev0, 4885de2) 及依赖；模型目录已就位
用法:
    python 03_vllm_load.py --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic
可选:
    --max-model-len 262144   (默认。原生 262144，无 rope_scaling/YaRN)
    --quantization kl3-compressed-xline   (默认。W8A8 量化模式)
    --context-len 8000       (长上下文测试拼凑的 token 数；256K 全量留 04 压测)
    --eager                  (加此参数则 enforce_eager=True，便于首次诊断)

权威参数（2026-08-16 定稿，勿改）:
    --tensor-parallel-size 8（TP=8 单实例，最终决策）
    --dtype float16（=config dtype 字段，计算域精度；权重存储为 INT8）
    --quantization kl3-compressed-xline
    --mamba-ssm-cache-dtype float16（PR 408 必需，Qwen3.5 GDN kernel 不支持混合精度）

输出: 加载是否成功 + 短生成示例 + 长上下文召回结果
"""
import argparse
from vllm import LLM, SamplingParams


def build_long_prompt(n_tokens):
    """拼一段足够长的中文文本（约 n_tokens），内含一条可被召回的事实"""
    fact = "昆仑芯 P800 的峰值 INT8 算力是 200 TOPS。"
    filler = ("今天的天气很好，阳光洒在操场上，学生们在课间休息，有人在打篮球，有人在聊天。"
              "远处的山峦连绵起伏，一条小河从山谷中流过，水声哗哗作响。")
    fact_tokens = 25  # 粗略估算
    fill_tokens = 60
    # 把事实放最前面，fill 填充后面，最后在结尾问事实
    body = fact + filler * max(1, (n_tokens - fact_tokens) // fill_tokens)
    q = "\n\n问题：根据上文，昆仑芯 P800 的峰值 INT8 算力是多少？请只回答数值和单位。"
    return body + q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型目录")
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--quantization", default="kl3-compressed-xline",
                    help="W8A8 量化模式（默认 kl3-compressed-xline）")
    ap.add_argument("--dtype", default="float16", help="=config dtype 字段；勿改 bfloat16（double VRAM bug）")
    ap.add_argument("--mamba-ssm-cache-dtype", default="float16", help="PR 408 必需")
    ap.add_argument("--context-len", type=int, default=8000)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=8,  # TP=8 单实例（用户最终决策，不传 --tp）
        dtype=args.dtype,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        enable_chunked_prefill=True,
        enforce_eager=args.eager,
        gpu_memory_utilization=0.9,
    )
    print("=" * 60)
    print("[OK] 模型加载成功  (TP=8, %s, %s, max_len=%d)" % (args.quantization, args.dtype, args.max_model_len))
    print("=" * 60)

    # 短生成
    short = SamplingParams(max_tokens=64, temperature=0.7)
    out = llm.generate(["你好，请用一句话介绍你自己。"], short)
    print("[短生成] ", out[0].outputs[0].text.strip()[:200])

    # 长上下文召回（验证长文不丢事实；原生 262144 无 YaRN）
    prompt = build_long_prompt(args.context_len)
    print(f"[长上下文] 构造了约 {args.context_len} token 的输入，开始生成...")
    long = SamplingParams(max_tokens=48, temperature=0.0)
    out2 = llm.generate([prompt], long)
    ans = out2[0].outputs[0].text.strip()
    print(f"[长上下文] 模型回答: {ans[:200]}")
    hit = "200" in ans
    print(f"[{'PASS' if hit else 'CHECK'}] 召回结果: {'包含正确数值 200' if hit else '未见 200，需人工判断是否长上下文失效'}")

    print("=" * 60)
    print("下一步: 跑 04 吞吐压测（短/长上下文分开测）")


if __name__ == "__main__":
    main()
