#!/bin/bash
# probe_rollback.sh — 回退 4885de2 前的环境探测（只读）
# 1) vllm_kunlun 安装形态（editable 链接 or 真实目录） 2) vllm 版本 3) patches 内容 4) 构建配置
SP=/opt/vllm_kunlun/lib/python3.10/site-packages
PY=/opt/vllm_kunlun/bin/python
echo "=== 1. site-packages 里 vllm 相关布局 ==="
ls -la $SP/ | grep -iE "vllm|editable"
echo ""
echo "=== 2. vllm 当前版本 ==="
$PY -c "import vllm; print(vllm.__version__)" 2>&1 | tail -2
echo ""
echo "=== 3. vllm_kunlun/patches 内容（4885de2 自带补丁对照）==="
ls $SP/vllm_kunlun/patches/ 2>/dev/null
echo ""
echo "=== 4. 构建配置（pyproject.toml / setup.py 头部）==="
head -25 $SP/vllm_kunlun/pyproject.toml 2>/dev/null
ls $SP/vllm_kunlun/setup.py 2>/dev/null && head -30 $SP/vllm_kunlun/setup.py
echo ""
echo "=== 5. editable 指向（如为链接）==="
if [ -L $SP/vllm_kunlun ]; then
  readlink -f $SP/vllm_kunlun
fi
echo ""
echo "=== 6. editable 映射目标 + import 实际加载路径 ==="
cat $SP/_editable_impl_vllm_kunlun.pth 2>/dev/null
echo ""
ls -la $SP/_editable_impl_vllm_kunlun.py 2>/dev/null
head -40 $SP/_editable_impl_vllm_kunlun.py 2>/dev/null
echo "--- repo root ---"
ls /home/newdata/vLLM-Kunlun-0.25.1-dev/ 2>/dev/null | head -15
echo "--- import real path ---"
$PY -c "import vllm_kunlun; print(vllm_kunlun.__file__)" 2>/dev/null
echo "--- import vllm version ---"
$PY -c "import vllm; print(vllm.__version__)" 2>/dev/null
echo "PROBE_DONE"
