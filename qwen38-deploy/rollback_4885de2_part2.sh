#!/bin/bash
# rollback_4885de2_part2.sh — 方案一 阶段 2：编译 _kunlun + 4885de2 自带补丁 + import 验证
# 在 part1 之后运行；幂等（编译产物存在则跳过）
set -u
SP=/opt/vllm_kunlun/lib/python3.10/site-packages
PY=/opt/vllm_kunlun/bin/python
TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/newdata/logs/rollback_p2_$TS.log
mkdir -p /home/newdata/logs
exec > >(tee -a $LOG) 2>&1

echo "===== part2 开始 $TS ====="
cd /tmp/vllm_kunlun_4885de2 || { echo "NO_EXTRACT_DIR（先跑 part1）"; exit 1; }

echo "=== 1. 4885de2 patches 结构 ==="
ls vllm_kunlun/patches/
ls vllm_kunlun/quantization/ 2>/dev/null | head -5

echo "=== 2. 编译 _kunlun ==="
if [ -d build ] && [ -f build/lib.linux-x86_64-cpython-310/vllm_kunlun/_C/_kunlun*.so ]; then
  echo "BUILD_ALREADY_EXISTS（跳过）"
else
  python setup.py build 2>&1 | tail -8
  echo "BUILD_STAGE_DONE（exit=$?）"
fi

echo "=== 3. 安装扩展 ==="
python setup.py install 2>&1 | tail -5
ls $SP/vllm_kunlun/_C/*.so 2>/dev/null && echo "SO_INSTALLED"

echo "=== 4. 4885de2 自带补丁 ==="
# eval_frame（torch 2.5.1 时代官方补丁）
cp vllm_kunlun/patches/eval_frame.py $SP/torch/_dynamo/eval_frame.py && echo "EVAL_FRAME_OK"
# quantization 注册表补丁（官方流程④：包内 quantization/__init__.py → vllm 的量化注册表）
if [ -f vllm_kunlun/quantization/__init__.py ]; then
  cp vllm_kunlun/quantization/__init__.py $SP/vllm/model_executor/layers/quantization/__init__.py && echo "QUANT_OK"
else
  echo "QUANT_NA（4885de2 无 quantization 补丁）"
fi
# patch_torch251（对 vllm 0.15.x 写的，应高比例适用）
python vllm_kunlun/patches/patch_torch251.py 2>&1 | tail -15
echo "PATCHES_DONE"

echo "=== 5. import 验证 ==="
cd /tmp
timeout 180 $PY -c "import vllm; from vllm import LLM; print('VLLM_IMPORT_OK', vllm.__version__)" 2>&1 | grep -vE "^\s*INFO|XCCL|SYMBOL_REWRITE" | tail -15

echo "=== 6. vllm_kunlun import + register 验证 ==="
timeout 180 $PY -c "import vllm_kunlun; from vllm_kunlun import register; register(); print('KUNLUN_REGISTER_OK')" 2>&1 | grep -vE "^\s*INFO|XCCL|SYMBOL_REWRITE" | tail -10

echo "===== part2 完成 $TS ====="
