#!/bin/bash
# c_step.sh — C 阶段: 从 serve 日志提取 step 级耗时分布（decode/prefill 分类统计）
L=/home/newdata/logs/serve_20260816_180850.log
# Iteration 行格式: Iteration(N): X context requests, Y context tokens, Z generation requests, W generation tokens, iteration elapsed time: T ms
grep "Iteration(" "$L" | sed -E 's/.*Iteration\(([0-9]+)\): ([0-9]+) context requests, ([0-9]+) context tokens, ([0-9]+) generation requests, ([0-9]+) generation tokens, iteration elapsed time: ([0-9.]+) ms/\1 \2 \3 \4 \5 \6/' > /tmp/steps.txt
echo "== step 总数 =="
wc -l < /tmp/steps.txt
echo
echo "== decode-only steps（generation requests>0 且 context=0）: 耗时分布 =="
awk '$4>0 && $2==0 {print $6}' /tmp/steps.txt | sort -n | awk '{a[NR]=$1} END {print "count="NR; if(NR>0){print "min="a[1]; print "p50="a[int(NR*0.5)]; print "p90="a[int(NR*0.9)]; print "max="a[NR]; s=0; for(i=1;i<=NR;i++) s+=a[i]; print "mean="s/NR}}'
echo
echo "== decode-only: 按 generation tokens 分组耗时均值 =="
awk '$4>0 && $2==0 {sum[$5]+=$6; n[$5]++; g[$5]=$4} END {for(k in g) printf "batch=%d tokens=%d n=%d mean=%.1fms\n", k, g[k], n[k], sum[k]/n[k]}' /tmp/steps.txt | sort -t= -k2 -n | head -20
echo
echo "== prefill-only steps（context>0 且 generation=0）: 耗时与 tok/s =="
awk '$4==0 && $2>0 {sum+=$6; tok+=$3; n++; if($6>max) max=$6} END {print "count="n" total_tokens="tok" total_ms="sum; if(sum>0) printf "avg %.1f ms/step, %.1f input tok/s\n", sum/n, tok*1000/sum; printf "max step %.0f ms\n", max}' /tmp/steps.txt
echo
echo "== 最后 60 行原始样例（05 压测时段） =="
tail -60 "$L" | grep "Iteration(" | tail -8
