#!/usr/bin/env python3
# concurrency_bench.py — 测 Dense 多并发总吞吐曲线（API 层）
import time
import json
import urllib.request
import concurrent.futures as cf

HOST = "127.0.0.1"
PORT = 8000
MODEL = "qwen3.8-kunlun"
KEY = open("/home/newdata/qwen38-deploy/api_key.txt").read().strip()

PROMPT = "The quick brown fox jumps over the lazy dog. " * 12  # ~172 tok
MAX_TOKENS = 256


def one_req(_):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.loads(resp.read())
    dt = time.monotonic() - t0
    ct = d.get("usage", {}).get("completion_tokens", 0)
    return ct, dt


for n in [1, 4, 8, 16, 32, 64, 128, 256]:
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(one_req, range(n)))
    wall = time.monotonic() - t0
    total_tok = sum(ct for ct, _ in results)
    max_dt = max(dt for _, dt in results)
    throughput = total_tok / wall
    per_req = total_tok / n / max_dt
    print(
        f"并发={n:3d}: 总吞吐={throughput:7.1f} tok/s, "
        f"单请求={per_req:5.1f} tok/s, 墙钟={wall:.1f}s, 总tok={total_tok}"
    )
