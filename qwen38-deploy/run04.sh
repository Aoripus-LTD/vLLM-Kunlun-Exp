#!/bin/bash
# run04.sh — 04 吞吐压测（short + long，仅 TP=8；2026-08-16 用户批准）
# 用法：docker exec -d qwen38-p800 bash /home/newdata/run04.sh
set -u
TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/newdata/logs/04_bench_$TS.log
mkdir -p /home/newdata/logs
source /home/newdata/vLLM-Kunlun-0.25.1-dev/setup_env.sh
export PATH=/opt/vllm_kunlun/bin:$PATH
cd /home/newdata
echo "===== RUN04 开始 $TS =====" | tee $LOG
echo "--- short 组（input 512 / output 512, 256 prompts）---" | tee -a $LOG
/opt/vllm_kunlun/bin/python 04_throughput_bench.py \
  --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic \
  --group short --output-json bench_short.json 2>&1 | tee -a $LOG
SHORT_EXIT=${PIPESTATUS[0]}
echo "SHORT_EXIT=$SHORT_EXIT" | tee -a $LOG
echo "--- long 组（input 32768 / output 256, 16 prompts）---" | tee -a $LOG
/opt/vllm_kunlun/bin/python 04_throughput_bench.py \
  --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic \
  --group long --num-prompts 16 --output-json bench_long.json 2>&1 | tee -a $LOG
LONG_EXIT=${PIPESTATUS[0]}
echo "LONG_EXIT=$LONG_EXIT" | tee -a $LOG
echo "===== RUN04 完成 $TS =====" | tee -a $LOG
echo "RUN04_EXIT=$((SHORT_EXIT + LONG_EXIT))" | tee -a $LOG
