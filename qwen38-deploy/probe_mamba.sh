#!/bin/bash
# probe_mamba.sh — 探查 mamba_utils 调用条件 + vllm_kunlun 覆盖情况
P=/opt/vllm_kunlun/lib/python3.10/site-packages
echo "== gpu_model_runner.py 3395-3430（preprocess_mamba 调用条件） =="
sed -n '3395,3430p' $P/vllm/v1/worker/gpu_model_runner.py
echo
echo "== mamba_utils.py 1-200（batch_memcpy 与 preprocess_mamba 逻辑） =="
sed -n '1,200p' $P/vllm/v1/worker/mamba_utils.py
echo
echo "== vllm_kunlun/v1/worker/ 目录 =="
ls $P/vllm_kunlun/v1/worker/ 2>/dev/null || echo "无 vllm_kunlun/v1/worker/"
echo
echo "== 04 日志是否含 mamba 相关 =="
grep -c -i "mamba" /home/newdata/logs/04_bench_153510.log 2>/dev/null || echo "04 日志不存在"
