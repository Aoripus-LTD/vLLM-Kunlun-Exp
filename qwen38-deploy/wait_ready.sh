#!/bin/bash
# wait_ready.sh — 轮询 vllm serve /v1/models 就绪（容器内执行）
# 用法: bash wait_ready.sh <serve 日志路径>
LOG=${1:-/home/newdata/logs/serve_latest.log}
for i in $(seq 1 36); do
  if curl -s --max-time 5 http://127.0.0.1:8000/v1/models | grep -q models; then
    echo "READY after $((i * 10))s"
    curl -s http://127.0.0.1:8000/v1/models
    exit 0
  fi
  sleep 10
done
echo "NOT_READY in 360s"
tail -30 "$LOG"
exit 1
