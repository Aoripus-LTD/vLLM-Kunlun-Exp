#!/bin/bash
# rollback_4885de2_part3.sh — 修复 part2 路径 bug：仓库根编译 + site-packages 内取补丁
# 用户批准 2026-08-16（方案一）
set -u
SP=/opt/vllm_kunlun/lib/python3.10/site-packages
PY=/opt/vllm_kunlun/bin/python
REPO=/home/newdata/vLLM-Kunlun-0.25.1-dev
TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/newdata/logs/rollback_p3_$TS.log
mkdir -p /home/newdata/logs
exec > >(tee -a $LOG) 2>&1

echo "===== part3 开始 $TS ====="
echo "=== 0. 确认 4885de2 包内 patches/quantization 结构 ==="
ls $SP/vllm_kunlun/patches/ 2>/dev/null || echo "NO_PATCHES_IN_PKG"
ls $SP/vllm_kunlun/quantization/ 2>/dev/null | head -5 || echo "NO_QUANT_MODULE"

echo "=== 1. 覆盖仓库根构建文件为 4885de2 版 ==="
cp /tmp/vllm_kunlun_4885de2/setup.py $REPO/setup.py
cp /tmp/vllm_kunlun_4885de2/pyproject.toml $REPO/pyproject.toml
cp /tmp/vllm_kunlun_4885de2/requirements.txt $REPO/requirements.txt
echo "BUILD_FILES_OK"

echo "=== 2. 编译 _kunlun（仓库根）==="
cd $REPO
if ls $SP/vllm_kunlun/_C/_kunlun*.so >/dev/null 2>&1; then
  echo "SO_ALREADY_INSTALLED"
else
  python setup.py build 2>&1 | tail -40
  python setup.py install 2>&1 | tail -10
fi
ls $SP/vllm_kunlun/_C/*.so 2>/dev/null && echo "SO_INSTALLED" || echo "SO_MISSING"

echo "=== 3. 4885de2 补丁（从 site-packages 包内取）==="
if [ -f $SP/vllm_kunlun/patches/eval_frame.py ]; then
  cp $SP/vllm_kunlun/patches/eval_frame.py $SP/torch/_dynamo/eval_frame.py && echo "EVAL_FRAME_OK"
fi
if [ -f $SP/vllm_kunlun/quantization/__init__.py ]; then
  cp $SP/vllm_kunlun/quantization/__init__.py $SP/vllm/model_executor/layers/quantization/__init__.py && echo "QUANT_OK"
fi
if [ -f $SP/vllm_kunlun/patches/patch_torch251.py ]; then
  python $SP/vllm_kunlun/patches/patch_torch251.py 2>&1 | tail -15
fi
echo "PATCHES_DONE"

echo "=== 4. import 验证（_kunlun 必须加载成功）==="
cd /tmp
timeout 180 $PY -c "import vllm; from vllm import LLM; print('VLLM_IMPORT_OK', vllm.__version__)" 2>&1 | grep -vE "^\s*INFO|XCCL|SYMBOL_REWRITE|UserWarning|pkg_resources" | tail -12

echo "===== part3 完成 $TS ====="
