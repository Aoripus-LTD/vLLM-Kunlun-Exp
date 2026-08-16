#!/bin/bash
# run03.sh — 03 加载验证（回退 4885de2 后第 1 次运行）
# 用 docker exec -d 后台执行；日志 /home/newdata/logs/03_rollback1.log
source /home/newdata/vLLM-Kunlun-0.25.1-dev/setup_env.sh
cd /home/newdata
mkdir -p /home/newdata/logs
/opt/vllm_kunlun/bin/python 03_vllm_load.py --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic > /home/newdata/logs/03_rollback1.log 2>&1
echo "RUN03_EXIT=$?" >> /home/newdata/logs/03_rollback1.log
