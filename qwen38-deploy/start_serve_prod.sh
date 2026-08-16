#!/bin/bash
# start_serve_prod.sh — Qwen3.8 Dense 正式生产服务启动脚本（容器内执行）
# 相比 start_serve.sh 新增：
#   --api-key（从 api_key.txt 读取，随机 64 hex 字符）
#   --served-model-name qwen3.8-kunlun（OpenAI API 模型名）
#   --max-num-seqs 256（最大并发序列数，超量请求排队等待）
set -e
cd /home/newdata
source /home/newdata/vLLM-Kunlun-0.25.1-dev/setup_env.sh
export VLLM_USE_V1=1
export VLLM_HOST_IP=$(hostname -i)

API_KEY=$(cat /home/newdata/qwen38-deploy/api_key.txt)

TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/newdata/logs/serve_prod_${TS}.log
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
  --api-key "$API_KEY" \
  --served-model-name qwen3.8-kunlun \
  --max-num-seqs 256 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  > "$LOG" 2>&1 &

echo $! > /home/newdata/logs/serve.pid
echo "[start_serve_prod] pid=$(cat /home/newdata/logs/serve.pid) log=$LOG"
