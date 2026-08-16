#!/bin/bash
# fix_conv1d_spec.sh — GDN 层 speculative conv_state 更新修复（幂等）
# 用法：容器内 bash fix_conv1d_spec.sh
set -e
P=/opt/vllm_kunlun/lib/python3.10/site-packages
F=$P/vllm_kunlun/ops/mamba/causal_conv1d.py
PY=/opt/vllm_kunlun/bin/python
DIR=/home/newdata/qwen38-deploy

echo "== 1. 备份 =="
B=$F.bak_conv1d_spec
if [ ! -f "$B" ]; then
  cp "$F" "$B"
  echo "备份 -> $B"
else
  echo "备份已存在: $B"
fi
md5sum "$F" "$B"

echo
echo "== 2. 应用补丁（幂等：先恢复再 apply） =="
cp "$B" "$F"
$PY "$DIR/fix_conv1d_spec.py" --apply "$F"

echo
echo "== 3. py_compile 校验 =="
$PY -m py_compile "$F" && echo "py_compile OK"

echo
echo "== 4. 单测 =="
$PY "$DIR/fix_conv1d_spec.py" --test "$P"

echo
echo "== 5. 最终状态 =="
md5sum "$F"
grep -n "num_accepted_tokens\[i\]" "$F" | head -3
echo "ALL DONE"
