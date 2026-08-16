#!/bin/bash
# c_ttft.sh — 提取 05 冷热轮 TTFT 对比数据
grep -E "\[cold\]|\[hot\]|TTFT|对比|round|冷" /home/newdata/logs/05_prefix_bench2.log | head -40
