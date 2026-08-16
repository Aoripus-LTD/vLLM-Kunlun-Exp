#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch_docs.py — 爬取 vllm-kunlun readthedocs 全站做本地索引

用法:  python fetch_docs.py [输出目录]
默认输出: D:\\Workspace\\vllm-kunlun-docs
限制: 最多 600 页，仅 /en/stable/ 下同源 HTML（本地 Grep 可全文检索）
"""
import os
import re
import sys
import time
import urllib.parse

import requests

BASE = "https://vllm-kunlun.readthedocs.io/en/stable/"
OUT = sys.argv[1] if len(sys.argv) > 1 else r"D:\Workspace\vllm-kunlun-docs"
HDRS = {"User-Agent": "Mozilla/5.0 (docs-mirror; local-index-build)"}
MAX_PAGES = 600
BAD_CHARS = re.compile(r'[<>:"|?*]')


def norm(u):
    """仅保留 vllm-kunlun.readthedocs.io /en/stable/ 下的同源 HTML"""
    p = urllib.parse.urlparse(u)
    if p.netloc != "vllm-kunlun.readthedocs.io":
        return None
    if not p.path.startswith("/en/stable/"):
        return None
    if not (p.path.endswith(".html") or p.path.endswith("/")):
        return None
    return urllib.parse.urlunparse((p.scheme, p.netloc, p.path, "", "", ""))


def safe_name(s):
    return BAD_CHARS.sub("_", s)


def main():
    os.makedirs(OUT, exist_ok=True)
    seen, todo = set(), [BASE]
    n = 0
    while todo and n < MAX_PAGES:
        u = todo.pop(0)
        if u in seen:
            continue
        seen.add(u)
        try:
            r = requests.get(u, headers=HDRS, timeout=30)
            if r.status_code != 200:
                print(f"[skip {r.status_code}] {u}")
                continue
        except Exception as e:
            print(f"[err] {u}: {e}")
            continue
        p = urllib.parse.urlparse(u)
        rel = p.path[len("/en/stable/"):] or "index"
        if rel.endswith("/"):
            rel += "index.html"
        elif not rel.endswith(".html"):
            rel += ".html"
        fp = os.path.join(OUT, safe_name(rel).replace("/", os.sep))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(r.text)
        n += 1
        for m in re.findall(r'href="([^"#]+)"', r.text):
            nu = norm(urllib.parse.urljoin(u, m))
            if nu:
                todo.append(nu)
        time.sleep(0.1)
    print(f"[done] {n} pages -> {OUT}")


if __name__ == "__main__":
    main()
