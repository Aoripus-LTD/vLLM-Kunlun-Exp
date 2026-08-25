import time
import json
import urllib.request

MODEL = "/home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic"
KEY = open("/home/newdata/qwen38-deploy/api_key.txt").read().strip()
PROMPT = "The quick brown fox jumps over the lazy dog. " * 12

body = {
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 32,
    "temperature": 0.0,
    "stream": True,
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
)
t0 = time.monotonic()
deltas = []
prev = t0
first = None
with urllib.request.urlopen(req, timeout=600) as resp:
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            break
        now = time.monotonic()
        if first is None:
            first = now
        deltas.append(now - prev)
        prev = now
total = time.monotonic() - t0
n = len(deltas)
deltas_sorted = sorted(deltas)
print(f"tokens={n} ttft={first-t0:.3f}s total={total:.3f}s overall={n/total:.1f} tok/s")
print(f"interval: min={deltas_sorted[0]*1000:.1f} p50={deltas_sorted[n//2]*1000:.1f} p90={deltas_sorted[int(n*0.9)]*1000:.1f} max={deltas_sorted[-1]*1000:.1f} ms")
print("first 8:", [f"{d*1000:.1f}" for d in deltas[:8]], "ms")
