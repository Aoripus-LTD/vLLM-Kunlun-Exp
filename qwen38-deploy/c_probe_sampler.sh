#!/bin/bash
# c_probe_sampler.sh — 查 TopKTopPSampler 构造签名
P=/opt/vllm_kunlun/lib/python3.10/site-packages/vllm_kunlun/v1/sample/ops/topk_topp_sampler.py
sed -n '1,50p' "$P"
echo "---- forward_kunlun 签名 ----"
grep -n -A25 "def forward_kunlun" "$P" | head -30
