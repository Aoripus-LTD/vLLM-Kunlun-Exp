#!/bin/bash
# fix_speculative_qwen35_v2.sh — hf_config_override 就地提升 text_config 字段（第 17 雷 v2）
# v1（重建 Qwen3NextConfig）的返回值被 ModelConfig 调用方丢弃（hf_overrides 是就地修改
# 回调），vocab_size 等字段未落在 drafter config 上 → Qwen3NextMultiTokenPredictor
# AttributeError。v2 改为就地 setattr 提升，无返回值依赖。
# 用法：容器内 bash fix_speculative_qwen35_v2.sh
set -e
P=/opt/vllm_kunlun/lib/python3.10/site-packages
F=$P/vllm/config/speculative.py
PY=/opt/vllm_kunlun/bin/python
DIR=/home/newdata/qwen38-deploy

echo "== 1. 备份 =="
B=$F.bak_mtp_v2
if [ ! -f "$B" ]; then
  cp "$F" "$B"
  echo "备份 -> $B"
else
  echo "备份已存在: $B"
fi
md5sum "$F" "$B"

echo
echo "== 2. 恢复 v1 状态并重新应用补丁（幂等） =="
cp "$B" "$F"
$PY "$DIR/fix_speculative_qwen35_v2.py" --apply "$F"

echo
echo "== 3. py_compile 校验 =="
$PY -m py_compile "$F" && echo "py_compile OK"

echo
echo "== 4. 单测（hf_config_override 对真实 config 的 10 项断言） =="
$PY "$DIR/fix_speculative_qwen35_v2.py" --test "$P"

echo
echo "== 5. 最终状态 =="
md5sum "$F"
grep -n "qwen3_5\", \"qwen3_5_text\|not hasattr(hf_config, k)" "$F" | head -5
echo "ALL DONE"
