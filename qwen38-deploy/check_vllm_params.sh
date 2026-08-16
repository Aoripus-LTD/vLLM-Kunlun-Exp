#!/bin/bash
# check_vllm_params.sh — vllm 0.15.1 参数存在性检查（只读）
SP=/opt/vllm_kunlun/lib/python3.10/site-packages
PY=/opt/vllm_kunlun/bin/python
echo "=== 1. config.py 中 mamba_ssm_cache_dtype / enable_chunked_prefill ==="
grep -n "mamba_ssm_cache_dtype\|enable_chunked_prefill" $SP/vllm/config.py | head -8
echo ""
echo "=== 2. config 文件位置（0.15.1 结构变化探测）==="
ls $SP/vllm/config* 2>/dev/null
ls $SP/vllm/config/ 2>/dev/null | head -10
echo ""
echo "=== 3. LLM.__init__ 签名（文件定位）==="
ls $SP/vllm/entrypoints/llm.py 2>/dev/null
find $SP/vllm -maxdepth 2 -name "llm.py" 2>/dev/null | head -5
echo ""
echo "=== 4. CacheConfig 字段直测 ==="
timeout 120 $PY -c "from vllm.config import CacheConfig; fs = getattr(CacheConfig, '__dataclass_fields__', None); print('HAS_FIELDS:', fs is not None); print('mamba_ssm_cache_dtype' in (fs or {})); print('enable_chunked_prefill' in (fs or {}))" 2>&1 | grep -vE "^\s*INFO|XCCL|SYMBOL_REWRITE|UserWarning|pkg_resources" | tail -5
echo "CHECK_DONE"
