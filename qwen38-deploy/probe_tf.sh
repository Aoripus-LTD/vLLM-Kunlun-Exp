#!/bin/bash
# probe_tf.sh — transformers 版本 + Qwen2VLImageProcessor.max_pixels 探测（只读）
SP=/opt/vllm_kunlun/lib/python3.10/site-packages
PY=/opt/vllm_kunlun/bin/python
echo "=== 1. transformers 版本 ==="
timeout 60 $PY -c "import transformers; print(transformers.__version__)" 2>&1 | grep -vE "^\s*INFO|XCCL|SYMBOL_REWRITE|UserWarning|pkg_resources" | tail -2
echo ""
echo "=== 2. Qwen2VLImageProcessor 定义中 max_pixels ==="
grep -n "max_pixels" $SP/transformers/models/qwen2_vl/image_processing_qwen2_vl.py | head -8
echo ""
echo "=== 3. vllm 0.15.1 qwen2_vl.py:900-945 上下文 ==="
sed -n '900,945p' $SP/vllm/model_executor/models/qwen2_vl.py
echo ""
echo "=== 4. 已装 transformers dist-info ==="
ls -d $SP/transformers-*.dist-info 2>/dev/null
echo "PROBE_TF_DONE"
