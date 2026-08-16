#!/bin/bash
# grep_diag.sh — 定位 serve 日志中 EngineCore 失败根因
LOG=${1:-/home/newdata/logs/serve_latest.log}
echo "== 第一个 Traceback 及上下文 =="
grep -n -m1 -A 35 "Traceback" "$LOG" | head -50
echo
echo "== ERROR/failed 行（前 30） =="
grep -n -E "ERROR|Error|failed|Failed" "$LOG" | head -30
