#!/bin/bash
# start_serve_mtp_prof.sh — MTP + torch profiler（临时 profiling 用）
set -e
cd /home/newdata
source /home/newdata/vLLM-Kunlun-0.25.1-dev/setup_env.sh
export VLLM_USE_V1=1
export VLLM_HOST_IP=$(hostname -i)
export XMLIR_DYNAMO_WORKAROUND=1

echo "== 停止旧服务 =="
if [ -f /home/newdata/logs/serve.pid ]; then
  OLD_PID=$(cat /home/newdata/logs/serve.pid)
  kill $OLD_PID 2>/dev/null || true
  sleep 3
  kill -9 $OLD_PID 2>/dev/null || true
fi
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2
rm -f /home/newdata/logs/serve.pid

TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/newdata/logs/serve_mtp_prof_${TS}.log
mkdir -p /home/newdata/logs
rm -rf /home/newdata/logs/mtp_prof

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
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/newdata/logs/mtp_prof"}' \
  > "$LOG" 2>&1 &

echo $! > /home/newdata/logs/serve.pid
echo "[start_serve_mtp_prof] pid=$! log=$LOG"
