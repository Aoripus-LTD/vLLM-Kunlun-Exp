#!/bin/bash
# stop_serve.sh — 停止 vllm serve 服务（容器内执行）
# 读 serve.pid 优雅停止，5s 后 KILL 兜底，再 pkill 清理残留
PID_FILE=/home/newdata/logs/serve.pid

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  echo "[stop_serve] 发送 TERM 到 pid=$PID"
  kill "$PID" 2>/dev/null || true
  sleep 5
  kill -9 "$PID" 2>/dev/null || true
  rm -f "$PID_FILE"
else
  echo "[stop_serve] 无 serve.pid，跳过"
fi

# 兜底：清理残留 api_server 进程
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
echo "[stop_serve] done"
