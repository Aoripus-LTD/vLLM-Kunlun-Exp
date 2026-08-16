#!/bin/bash
# rollback_4885de2_part1.sh — 方案一 阶段 1：备份 + 替换源码 + vllm 0.25.1→0.15.1
# 用户批准 2026-08-16（方案一：回退官方验证组合 4885de2）
# 前提：/home/newdata/vllm_kunlun_4885de2.tar.gz 已上传
set -u
SP=/opt/vllm_kunlun/lib/python3.10/site-packages
PY=/opt/vllm_kunlun/bin/python
UV=/root/.local/bin/uv
TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/newdata/logs/rollback_p1_$TS.log
mkdir -p /home/newdata/logs /home/newdata/backup
exec > >(tee -a $LOG) 2>&1

echo "===== part1 开始 $TS ====="

echo "=== 0. 前置检查 ==="
[ -f /home/newdata/vllm_kunlun_4885de2.tar.gz ] || { echo "NO_TARBALL"; exit 1; }

echo "=== 1. 备份现状（2fda97b + vllm 版本）==="
tar -czf /home/newdata/backup/vllm_kunlun_2fda97b_$TS.tar.gz -C $SP vllm_kunlun
$PY -c "import vllm; print(vllm.__version__)" 2>/dev/null > /home/newdata/backup/vllm_ver_before_$TS.txt
ls -la /home/newdata/backup/ | grep -E "2fda97b|vllm_ver" | tail -2

echo "=== 2. 解压 4885de2 ==="
cd /tmp && rm -rf vllm_kunlun_4885de2 && mkdir vllm_kunlun_4885de2 && cd vllm_kunlun_4885de2
tar -xzf /home/newdata/vllm_kunlun_4885de2.tar.gz
echo "EXTRACT_OK"
ls setup.py pyproject.toml requirements.txt vllm_kunlun/ | head -8

echo "=== 3. 替换 site-packages/vllm_kunlun → 4885de2 ==="
rm -rf $SP/vllm_kunlun
mv /tmp/vllm_kunlun_4885de2/vllm_kunlun $SP/vllm_kunlun
echo "REPLACE_OK"
# 同步仓库根（pth 映射目标，防旧文件残留）
rm -rf /home/newdata/vLLM-Kunlun-0.25.1-dev/vllm_kunlun
cp -r $SP/vllm_kunlun /home/newdata/vLLM-Kunlun-0.25.1-dev/vllm_kunlun
echo "SRC_SYNC_OK"

echo "=== 4. 清理 vllm 手动补丁残留 + 重装 vllm 0.15.1（--no-deps）==="
rm -f $SP/vllm/model_executor/layers/quantization/__init__.py.bak_* 2>/dev/null || true
# 手改过/拷入的 quantization 补丁先删（uv 卸载不删未记录文件，防残留与 0.15.1 冲突）
ls $SP/vllm/model_executor/layers/quantization/__init__.py 2>/dev/null && echo "QUANT_FILE_EXISTS(BEFORE)"
$UV pip install --python $PY --index-url https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall --no-deps vllm==0.15.1
echo "VLLM_REINSTALL_DONE"
$PY -c "import vllm; print(vllm.__version__)" 2>/dev/null | grep -E "^[0-9]" | tail -1

echo "===== part1 完成 $TS ====="
