#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""watch_03_run.py — 轮询容器 03_run.log 直到成功/失败标记（paramiko，不碰 vllm）
用法: $env:QWEN38_SSH_PASS='密码'; python watch_03_run.py [轮数=120]
输出: MARKER: SUCCESS/FAILED (round N) + 关键日志行尾部
"""
import os
import re
import sys
import time
import paramiko

HOST = os.environ.get("QWEN38_HOST", "")
PORT = int(os.environ.get("QWEN38_PORT", "22"))
USER = os.environ.get("QWEN38_USER", "root")
PW = os.environ.get("QWEN38_SSH_PASS", "")
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
LOG = "/home/newdata/03_run.log"

OK_RE = re.compile(r"\[OK\] 模型加载成功|\[PASS\]|\[CHECK\]")
FAIL_RE = re.compile(
    r"Traceback|AssertionError|WorkerProc hit an exception|"
    r"Engine core initialization failed|Killed"
)


def ssh(cmd, timeout=60):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PW, timeout=30)
    try:
        _, so, se = c.exec_command(cmd, timeout=timeout)
        out = so.read().decode("utf-8", "replace")
        err = se.read().decode("utf-8", "replace")
        return out, err
    finally:
        c.close()


def main():
    for i in range(1, ROUNDS + 1):
        out, _ = ssh(f"tail -c 300000 {LOG} 2>/dev/null")
        if OK_RE.search(out):
            print(f"MARKER: SUCCESS (round {i})")
            print("--- 日志尾部（grep 关键行）---")
            key, _ = ssh(
                f"grep -E '\\[OK\\]|\\[PASS\\]|\\[CHECK\\]' {LOG} | tail -8"
            )
            print(key)
            return
        if FAIL_RE.search(out):
            print(f"MARKER: FAILED (round {i})")
            print("--- 日志尾部（grep 关键行）---")
            key, _ = ssh(
                f"grep -E 'ERROR|Traceback|Error' {LOG} | "
                f"grep -vE 'Triton|pkg_resources|post-import' | tail -30"
            )
            print(key)
            return
        if i % 6 == 0:
            print(f"round {i}: RUNNING")
        time.sleep(10)
    print("MARKER: TIMEOUT (no marker found)")


if __name__ == "__main__":
    main()
