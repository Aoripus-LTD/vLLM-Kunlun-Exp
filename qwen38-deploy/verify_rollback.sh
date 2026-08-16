#!/bin/bash
# verify_rollback.sh — 回退后验证：_kunlun .so + import 链 + setup_env.sh 内容（只读）
SP=/opt/vllm_kunlun/lib/python3.10/site-packages
PY=/opt/vllm_kunlun/bin/python
echo "=== 1. .so 位置（包根 vs _C/）==="
ls -la $SP/vllm_kunlun/_kunlun*.so 2>/dev/null
ls -la $SP/vllm_kunlun/_C/*.so 2>/dev/null || echo "NO_SO_IN_UC"
echo ""
echo "=== 2. import _kunlun 直测 ==="
timeout 60 $PY -c "import vllm_kunlun._kunlun as k; print('KUNLUN_SO_OK', k)" 2>&1 | grep -vE "^\s*INFO|XCCL|SYMBOL_REWRITE|UserWarning|pkg_resources" | tail -5
echo ""
echo "=== 3. setup_env.sh 内容 ==="
cat /home/newdata/vLLM-Kunlun-0.25.1-dev/setup_env.sh 2>/dev/null || cat $SP/vllm_kunlun/../setup_env.sh 2>/dev/null || echo "NO_SETUP_ENV"
echo "VERIFY_DONE"
