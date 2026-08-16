#!/bin/bash
# c_log.sh — C 阶段素材: 查看最新 serve 日志 + step 级耗时行格式
ls -t /home/newdata/logs/ | head -5
echo "== 最新 serve 日志 =="
LATEST=$(ls -t /home/newdata/logs/serve_*.log | head -1)
echo "log=$LATEST"
echo "== step 级行样例 =="
grep -m5 -E "iteration|step" "$LATEST" | head -5
echo "== step 行总数 =="
grep -c -E "iteration" "$LATEST"
