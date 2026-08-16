#!/bin/bash
# check_llm_sig.sh — vllm 0.15.1 LLM() 关键参数存在性检查（只读）
PY=/opt/vllm_kunlun/bin/python
timeout 120 $PY -c "from vllm import LLM; import inspect; s = inspect.signature(LLM.__init__).parameters; ks = ['mamba_ssm_cache_dtype','tensor_parallel_size','quantization','enable_chunked_prefill','enforce_eager','gpu_memory_utilization','max_model_len','dtype']; [print(k, k in s) for k in ks]" 2>&1 | grep -vE "^\s*INFO|XCCL|SYMBOL_REWRITE|UserWarning|pkg_resources" | tail -12
echo "SIG_CHECK_DONE"
