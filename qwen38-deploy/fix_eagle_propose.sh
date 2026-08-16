#!/bin/bash
# fix_eagle_propose.sh — propose 签名对齐 10 参数（幂等）
# 用法：容器内 bash fix_eagle_propose.sh
set -e
P=/opt/vllm_kunlun/lib/python3.10/site-packages
F=$P/vllm_kunlun/v1/sample/spec_decode/eagle.py
PY=/opt/vllm_kunlun/bin/python
DIR=/home/newdata/qwen38-deploy

echo "== 1. 备份 =="
B=$F.bak_eagle_propose
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
$PY "$DIR/fix_eagle_propose.py" --apply "$F"

echo
echo "== 3. py_compile 校验 =="
$PY -m py_compile "$F" && echo "py_compile OK"

echo
echo "== 4. 单测 =="
$PY "$DIR/fix_eagle_propose.py" --test "$P"

echo
echo "== 5. 最终状态 =="
md5sum "$F"
sed -n '30,42p' "$F"
echo "ALL DONE"
