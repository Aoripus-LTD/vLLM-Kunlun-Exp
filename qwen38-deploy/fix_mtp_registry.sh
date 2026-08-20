#!/bin/bash
# fix_mtp_registry.sh — 注册 Qwen3_5MTP 到 models/__init__.py + speculative.py
# draft 映射改指 Qwen3_5MTP（幂等）。用法：容器内 bash fix_mtp_registry.sh
set -e
P=/opt/vllm_kunlun/lib/python3.10/site-packages
PY=/opt/vllm_kunlun/bin/python
DIR=/home/newdata/qwen38-deploy

echo "== 1. models/__init__.py 注册 =="
F1=$P/vllm_kunlun/models/__init__.py
B1=$F1.bak_mtp_registry
if [ ! -f "$B1" ]; then
  cp "$F1" "$B1" && echo "备份 -> $B1"
else
  echo "备份已存在: $B1"
fi
$PY "$DIR/fix_mtp_registry.py" --apply-models "$F1"

echo
echo "== 2. speculative.py draft 映射 =="
F2=$P/vllm/config/speculative.py
B2=$F2.bak_mtp_registry
if [ ! -f "$B2" ]; then
  cp "$F2" "$B2" && echo "备份 -> $B2"
else
  echo "备份已存在: $B2"
fi
$PY "$DIR/fix_mtp_registry.py" --apply-spec "$F2"

echo
echo "== 3. 校验 =="
$PY "$DIR/fix_mtp_registry.py" --test "$P"
$PY -m py_compile "$F1" "$F2" && echo "py_compile OK"

echo
echo "== 4. import 验证 =="
$PY -c "
from vllm.model_executor.models import ModelRegistry
print('Qwen3_5MTP' in ModelRegistry.get_supported_archs(), 'Qwen3_5MTP registered')
"

echo "ALL DONE"
