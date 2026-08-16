#!/bin/bash
# c_step2.sh — C 阶段: decode step 完整分组（全 batch 值）+ 64 并发时段精确定位
L=/home/newdata/logs/serve_20260816_180850.log
grep "Iteration(" "$L" | sed -E 's/.*Iteration\(([0-9]+)\): ([0-9]+) context requests, ([0-9]+) context tokens, ([0-9]+) generation requests, ([0-9]+) generation tokens, iteration elapsed time: ([0-9.]+) ms/\1 \2 \3 \4 \5 \6/' > /tmp/steps.txt
echo "== decode-only 完整分组（按 generation tokens = batch） =="
awk '$4>0 && $2==0 {sum[$5]+=$6; n[$5]++; g[$5]=$4} END {for(k in g) printf "batch=%d n=%d mean=%.1fms  ->  %.0f tok/s/step\n", k, n[k], sum[k]/n[k], k*1000/(sum[k]/n[k])}' /tmp/steps.txt | sort -t= -k2 -n
echo
echo "== batch>=48 的 decode steps（接近满 batch 64） =="
awk '$4>=48 && $2==0 {print $5, $6}' /tmp/steps.txt | awk '{sum+=$2; n++; if($2>mx) mx=$2; if($2<mn||n==1) mn=$2} END {printf "n=%d mean=%.1fms min=%.1f max=%.1f -> %.0f tok/s\n", n, sum/n, mn, mx, 64*1000/(sum/n)}'
echo
echo "== 05 吞吐采样时段（18:16-18:17）的 decode step 时间线（每 5 个采样 1 行） =="
grep "Iteration(" "$L" | grep "18:1[67]" | sed -E 's/.*Iteration\(([0-9]+)\): [0-9]+ context requests, [0-9]+ context tokens, ([0-9]+) generation requests, [0-9]+ generation tokens, iteration elapsed time: ([0-9.]+) ms/step \1 gen=\2 time=\3ms/' | awk 'NR%5==1'
