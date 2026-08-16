#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""watch04.py — 04 压测日志轮询（paramiko，增量打印，检测 RUN04_EXIT 后退出）"""
import os
import sys
import time

import paramiko

LOG_GLOB = "/home/newdata/logs/04_bench_*.log"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(os.environ["QWEN38_HOST"], port=int(os.environ.get("QWEN38_PORT", "22")),
          username=os.environ.get("QWEN38_USER", "root"),
          password=os.environ["QWEN38_SSH_PASS"], timeout=30)

last_len = 0
while True:
    try:
        _, out, _ = c.exec_command(
            f"L=$(ls -t {LOG_GLOB} 2>/dev/null | head -1); if [ -n \"$L\" ]; then echo $L; cat $L; fi",
            timeout=60)
        data = out.read().decode(errors="replace")
        lines = data.splitlines()
        if lines and lines[0].startswith("/home/newdata/logs/04_bench_"):
            content = "\n".join(lines[1:])
            if len(content) > last_len:
                print(content[last_len:], end="", flush=True)
                last_len = len(content)
            if "RUN04_EXIT=" in content:
                for ln in content.splitlines():
                    if ln.startswith("RUN04_EXIT=") or "SHORT_EXIT=" in ln or "LONG_EXIT=" in ln:
                        print(f"WATCH04: {ln}", flush=True)
                sys.exit(0)
    except Exception as e:
        print(f"WATCH04 retry: {e}", flush=True)
    time.sleep(45)
