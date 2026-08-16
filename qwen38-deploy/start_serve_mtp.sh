#!/bin/bash
# start_serve_mtp.sh — MTP speculative decoding 服务启动（容器内执行）
# 相比 start_serve.sh 新增：
#   --speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}'
#   --enable-prefix-caching 已移除（第 13 雷：MambaModelConfig.verify_and_update_config
#   的断言 —— prefix caching 开启时 mamba_cache_mode 自动变 all/align，
#   "all" 触发 Qwen3NextForCausalLM/Qwen3NextMTP raise，"align" 与 speculative 断言冲突；
#   prefix caching 关闭 → mamba_cache_mode="none" 合法。即 vllm 0.15.1 限制：
#   MTP 与 mamba 前缀缓存不可共存）
# 前置条件：vllm/config/speculative.py 已打 qwen3_5 → qwen3_next 转换 patch（备份 .bak_mtp）
set -e
cd /home/newdata
source /home/newdata/vLLM-Kunlun-0.25.1-dev/setup_env.sh
export VLLM_USE_V1=1
export VLLM_HOST_IP=$(hostname -i)

echo "== 停止旧服务 =="
if [ -f /home/newdata/logs/serve.pid ]; then
  OLD_PID=$(cat /home/newdata/logs/serve.pid)
  echo "旧 pid=$OLD_PID，发送 TERM..."
  kill $OLD_PID 2>/dev/null || true
  sleep 3
  kill -9 $OLD_PID 2>/dev/null || true
fi
# 兜底：清理残留 api_server 进程
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2
echo "残留进程检查:"
ps aux | grep "vllm.entrypoints.openai" | grep -v grep || echo "  (无)"
rm -f /home/newdata/logs/serve.pid

TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/newdata/logs/serve_mtp_${TS}.log
mkdir -p /home/newdata/logs

echo "== 启动 MTP 服务 =="
nohup /opt/vllm_kunlun/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic \
  --tensor-parallel-size 8 \
  --dtype float16 \
  --quantization compressed-tensors \
  --max-model-len 262144 \
  --mamba-ssm-cache-dtype float16 \
  --gpu-memory-utilization 0.75 \
  --speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}' \
  --port 8000 \
  --enable-logging-iteration-details \
  > "$LOG" 2>&1 &

echo $! > /home/newdata/logs/serve.pid
echo "[start_serve_mtp] pid=$(cat /home/newdata/logs/serve.pid) log=$LOG"
