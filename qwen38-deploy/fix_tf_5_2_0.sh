#!/bin/bash
# fix_tf_5_2_0.sh — transformers 5.5.3 → 5.2.0（5.5.3 缺失 max_pixels API，vllm-kunlun 0.15.1.dev0 需要 5.2.0）
set -u
PY=/opt/vllm_kunlun/bin/python
UV=/root/.local/bin/uv
TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/newdata/logs/fix_tf_$TS.log
mkdir -p /home/newdata/logs
exec > >(tee -a $LOG) 2>&1
echo "===== fix_tf 开始 $TS ====="
echo "=== 1. 备份版本记录 ==="
timeout 60 $PY -c "import transformers; print(transformers.__version__)" 2>/dev/null > /home/newdata/backup/tf_ver_before_$TS.txt
cat /home/newdata/backup/tf_ver_before_$TS.txt
echo "=== 2. 降级 transformers==5.2.0（--no-deps）==="
$UV pip install --python $PY --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-deps --force-reinstall transformers==5.2.0
echo "=== 3. 验证 ==="
timeout 60 $PY -c "import transformers; print('TF_NOW:', transformers.__version__)" 2>&1 | grep -vE "^\s*INFO|XCCL|SYMBOL_REWRITE|UserWarning|pkg_resources" | tail -2
timeout 60 $PY -c "from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor; print('HAS_MAX_PIXELS:', hasattr(Qwen2VLImageProcessor, 'max_pixels'))" 2>&1 | grep -vE "^\s*INFO|XCCL|SYMBOL_REWRITE|UserWarning|pkg_resources" | tail -2
echo "===== fix_tf 完成 $TS ====="
