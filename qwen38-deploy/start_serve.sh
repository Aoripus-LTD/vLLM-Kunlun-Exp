#!/bin/bash
# start_serve.sh — vllm serve 正式服务启动脚本（容器内执行）
# 参数 = 03/04 权威参数 + prefix-caching + verbose；nohup 后台，日志 tee
set -e
cd /home/newdata
source /home/newdata/vLLM-Kunlun-0.25.1-dev/setup_env.sh
export VLLM_USE_V1=1
export VLLM_HOST_IP=$(hostname -i)

TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/newdata/logs/serve_${TS}.log
mkdir -p /home/newdata/logs

nohup /opt/vllm_kunlun/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic \
  --tensor-parallel-size 8 \
  --dtype float16 \
  --quantization compressed-tensors \
  --max-model-len 262144 \
  --mamba-ssm-cache-dtype float16 \
  --gpu-memory-utilization 0.75 \
  --enable-prefix-caching \
  --port 8000 \
  --enable-logging-iteration-details \
  > "$LOG" 2>&1 &

echo $! > /home/newdata/logs/serve.pid
echo "[start_serve] pid=$(cat /home/newdata/logs/serve.pid) log=$LOG"
