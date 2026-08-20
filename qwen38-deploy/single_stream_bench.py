#!/usr/bin/env python3
# single_stream_bench.py — 测 Dense 单流吞吐（TTFT + decode 稳态 tok/s）
import time
import urllib.request
import json

HOST = "127.0.0.1"
PORT = 8000
MODEL = "qwen3.8-kunlun"
KEY = open("/home/newdata/qwen38-deploy/api_key.txt").read().strip()

PROMPT = "The quick brown fox jumps over the lazy dog. " * 12  # ~108 tokens
MAX_TOKENS = 256


def stream():
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    t0 = time.monotonic()
    first_tok_time = None
    last_tok_time = None
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    if first_tok_time is None:
                        first_tok_time = time.monotonic()
                    last_tok_time = time.monotonic()
                    n_tokens += len(content) // 4  # 粗估 token 数（中文/英文混合不准，改用 completion_tokens 更好）
            except Exception:
                continue
    t_end = time.monotonic()
    return first_tok_time, last_tok_time, t0, t_end


# 用非流式精确拿 completion_tokens，再用流式拿时间
def nonstream():
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
    usage = d.get("usage", {})
    return usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0), dt


for trial in range(3):
    ct, pt, total_dt = nonstream()
    ft, lt, t0, te = stream()
    if ft is None or lt is None:
        print(f"trial{trial}: 无流式 token")
        continue
    ttft = ft - t0
    decode_dt = lt - ft
    # 流式 token 数用非流式的 completion_tokens
    decode_tok_s = (ct - 1) / decode_dt if decode_dt > 0 else 0
    overall = ct / total_dt
    print(
        f"trial{trial}: prompt={pt} tok, completion={ct} tok, "
        f"TTFT={ttft:.2f}s, decode={decode_tok_s:.1f} tok/s, "
        f"overall={overall:.1f} tok/s (total {total_dt:.2f}s)"
    )
