#!/bin/bash
# probe_mamba2.sh — 探查 copy_spec 结构（vllm_kunlun state_copy_func 实现）
P=/opt/vllm_kunlun/lib/python3.10/site-packages
echo "== vllm_kunlun/v1/worker/mamba_utils.py 全文 =="
cat $P/vllm_kunlun/v1/worker/mamba_utils.py
echo
echo "== vllm 原生 mamba_utils.py 200-280（postprocess_mamba 剩余） =="
sed -n '200,280p' $P/vllm/v1/worker/mamba_utils.py
echo
echo "== qwen3_5.py 中 state_copy_func / MambaStateCopyFunc =="
grep -rn -B3 -A25 "get_mamba_state_copy_func\|MambaStateCopyFunc" $P/vllm_kunlun/models/ 2>/dev/null | head -80
