#!/bin/bash
# sync_vllm_kunlun.sh — 容器内 vllm_kunlun 源码升级到 git 基线 2fda97b
# 前提: /home/newdata/vllm_kunlun_2fda97b.tar 已上传（本机 git archive 生成，LF）
set -e
SRC=/home/newdata/vLLM-Kunlun-0.25.1-dev
SP=/opt/vllm_kunlun/lib/python3.10/site-packages
PY=/opt/vllm_kunlun/bin/python
BK=/home/newdata/backup
TAR=/home/newdata/vllm_kunlun_2fda97b.tar

mkdir -p "$BK"

echo "== 1. 备份旧版 vllm_kunlun =="
BK_FILE="$BK/vllm_kunlun_old_$(date +%Y%m%d).tar.gz"
if [ ! -f "$BK_FILE" ]; then
  (cd "$SRC" && tar czf "$BK_FILE" vllm_kunlun)
fi
ls -lh "$BK"

echo "== 2. 替换源码（rm 旧 + 解压 2fda97b）=="
cd "$SRC"
rm -rf vllm_kunlun
tar xf "$TAR"
test -f vllm_kunlun/models/qwen3_5.py && echo "qwen3_5.py 存在 ✓"
test -d vllm_kunlun/models/deepseek_v4 && echo "deepseek_v4/ 存在 ✓"

echo "== 3. 重编译 _kunlun C++ 扩展 =="
"$PY" setup.py build
"$PY" setup.py install

echo "== 4. 重拷补丁到 site-packages =="
cp vllm_kunlun/patches/eval_frame.py "$SP/torch/_dynamo/eval_frame.py"
cp vllm_kunlun/quantization/__init__.py "$SP/vllm/model_executor/layers/quantization/__init__.py"
echo "补丁已重拷"

echo "== 5. 验证 vllm_kunlun 可导入 =="
"$PY" -c "import vllm_kunlun; print('vllm_kunlun import OK')"
echo "SYNC_DONE"
