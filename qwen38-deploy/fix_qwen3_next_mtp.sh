#!/bin/bash
# fix_qwen3_next_mtp.sh — 第 17 雷最终修复：MTP drafter 读方解包 Qwen3_5Config wrapper
# Qwen3NextMultiTokenPredictor / Qwen3NextMTP 读 vllm_config.model_config.hf_config，
# 该对象是 Qwen3_5Config 多模态 wrapper（字段在 text_config）→ vocab_size 等
# AttributeError。patch 读方：无 vocab_size 时解包 text_config。
# 用法：容器内 bash fix_qwen3_next_mtp.sh
set -e
P=/opt/vllm_kunlun/lib/python3.10/site-packages
F=$P/vllm/model_executor/models/qwen3_next_mtp.py
PY=/opt/vllm_kunlun/bin/python
DIR=/home/newdata/qwen38-deploy

echo "== 1. 备份 =="
B=$F.bak_mtp_unwrap
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
$PY "$DIR/fix_qwen3_next_mtp.py" --apply "$F"

echo
echo "== 3. py_compile 校验 =="
$PY -m py_compile "$F" && echo "py_compile OK"

echo
echo "== 4. 单测（真实 config 解包 6 项断言） =="
$PY "$DIR/fix_qwen3_next_mtp.py" --test "$P"

echo
echo "== 5. 最终状态 =="
md5sum "$F"
grep -n "text_cfg = getattr" "$F"
echo "ALL DONE"
