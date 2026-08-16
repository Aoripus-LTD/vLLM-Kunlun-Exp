#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ssh_run.py — SSH 执行/上传工具（paramiko）

用法:
    python ssh_run.py "命令..."               # 执行远程命令，stdout 回显
    python ssh_run.py --upload 本地文件 远程绝对路径   # 上传单个文件
    python ssh_run.py --upload-dir 本地目录 远程目录   # 递归上传目录

连接参数全部从环境变量读取（不入库）:
    QWEN38_HOST（必需）、QWEN38_PORT（默认 22）、QWEN38_USER（默认 root）、
    QWEN38_SSH_PASS（必需）。Windows PowerShell 示例:
    $env:QWEN38_HOST='...'; $env:QWEN38_SSH_PASS='密码'; python ssh_run.py "命令..."
"""
import os
import sys
import paramiko

HOST = os.environ.get("QWEN38_HOST", "")
PORT = int(os.environ.get("QWEN38_PORT", "22"))
USER = os.environ.get("QWEN38_USER", "root")
PW = os.environ.get("QWEN38_SSH_PASS", "")

if not HOST:
    sys.exit("[ssh_run] 缺少环境变量 QWEN38_HOST（服务器地址）")
if not PW:
    sys.exit("[ssh_run] 缺少环境变量 QWEN38_SSH_PASS（服务器密码）")


def _connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PW, timeout=30)
    return c


def run(cmd, timeout=300):
    """执行远程命令并回显 stdout/stderr，返回退出码"""
    c = _connect()
    try:
        _, so, se = c.exec_command(cmd, timeout=timeout)
        out = so.read().decode("utf-8", "replace")
        err = se.read().decode("utf-8", "replace")
        if out:
            print(out, end="")
        if err.strip():
            print("[stderr]", err, end="")
        return so.channel.recv_exit_status()
    finally:
        c.close()


def up(local, remote):
    c = _connect()
    try:
        s = c.open_sftp()
        rdir = os.path.dirname(remote)
        try:
            s.stat(rdir)
        except IOError:
            _, so, se = c.exec_command(f"mkdir -p '{rdir}'")
            so.read()
            se.read()
        s.put(local, remote)
        s.close()
        print(f"[upload] {local} -> {remote}")
    finally:
        c.close()


def up_dir(local_dir, remote_dir):
    c = _connect()
    try:
        s = c.open_sftp()
        _, so, se = c.exec_command(f"mkdir -p '{remote_dir}'")
        so.read()
        se.read()
        for root, _, files in os.walk(local_dir):
            rel = os.path.relpath(root, local_dir)
            rdir = remote_dir if rel == "." else f"{remote_dir}/{rel.replace(os.sep, '/')}"
            _, so, se = c.exec_command(f"mkdir -p '{rdir}'")
            so.read()
            se.read()
            for fn in files:
                lp = os.path.join(root, fn)
                rp = f"{rdir}/{fn}"
                s.put(lp, rp)
        s.close()
        print(f"[upload-dir] {local_dir} -> {remote_dir}")
    finally:
        c.close()


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        sys.exit(1)
    if a[0] == "--upload":
        up(a[1], a[2])
    elif a[0] == "--upload-dir":
        up_dir(a[1], a[2])
    else:
        sys.exit(run(" ".join(a)))
