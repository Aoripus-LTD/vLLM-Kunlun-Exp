#!/bin/bash
# hash_verify.sh — 容器内 vllm_kunlun 源码生成 md5 清单，与本机 2fda97b 基线比对
# 用法: bash hash_verify.sh   （生成 /home/newdata/vllm_kunlun_hashes.txt）
cd /home/newdata/vLLM-Kunlun-0.25.1-dev || exit 1
find vllm_kunlun -name "*.py" -not -path "*__pycache__*" -exec md5sum {} + | sort > /home/newdata/vllm_kunlun_hashes.txt
wc -l /home/newdata/vllm_kunlun_hashes.txt
echo "HASH_DONE"
