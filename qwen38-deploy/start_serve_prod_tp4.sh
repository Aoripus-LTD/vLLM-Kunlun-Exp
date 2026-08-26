#!/bin/bash
# start_serve_prod_tp4.sh — TP=4 降级生产（仅用健康卡 4/5/6/7）
# 背景：2026-08-26 XCCL 诊断确认卡 0/1/2/3 通信队列损坏，卡 4/5/6/7 正常。
set -e
cd /home/newdata
source /home/newdata/vLLM-Kunlun-0.25.1-dev/setup_env.sh
export VLLM_USE_V1=1
export VLLM_HOST_IP=$(hostname -i)
export XMLIR_DYNAMO_WORKAROUND=1
export XPU_SET_RECURRENT_GATED_DELTA_RULE_FWDV2_FP16_FAST_OPT=3
export CUDA_VISIBLE_DEVICES=4,5,6,7
unset XPU_VISIBLE_DEVICES

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
LOG=/home/newdata/logs/serve_prod_tp4_${TS}.log
mkdir -p /home/newdata/logs

nohup /opt/vllm_kunlun/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic \
  --tensor-parallel-size 4 \
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
echo "[start_serve_prod_tp4] pid=$(cat /home/newdata/logs/serve.pid) log=$LOG"
