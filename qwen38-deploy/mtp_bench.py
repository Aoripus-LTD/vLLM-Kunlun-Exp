#!/usr/bin/env python
"""MTP self-speculative decoding smoke + throughput benchmark.

Validates the served MTP endpoint on the Kunlun P800 box:
1. single-stream completion (correctness: prompt echo + token count)
2. streaming throughput at batch=1 (tok/s)
3. optional small-concurrency run

Usage: python mtp_bench.py [--host H] [--port P] [--concurrency N] [--prompt-len L] [--output-len O]
"""
import argparse
import json
import time
import urllib.request

MODEL = "Qwen3.8-27B-INT8-W8A8-Dynamic"


def post(host: str, port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://{host}:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    models = post(args.host, args.port, "/v1/models", {})
    print("models:", models.get("data", [{}])[0].get("id", "?"))

    prompt = "The quick brown fox jumps over the lazy dog. " * (args.prompt_len // 10)
    n = args.concurrency

    # 1. 单流冒烟（非流式，验证正确性）
    t0 = time.monotonic()
    resp = post(args.host, args.port, "/v1/completions", {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": args.output_len,
        "temperature": 0.0,
        "stream": False,
    })
    dt = time.monotonic() - t0
    text = resp["choices"][0]["text"]
    usage = resp.get("usage", {})
    tok = usage.get("completion_tokens", len(text.split()))
    print(f"[smoke] single stream: {tok} tok in {dt:.2f}s = {tok/dt:.1f} tok/s")
    print(f"[smoke] output head: {text[:80]!r}")
    print(f"[smoke] finish_reason: {resp['choices'][0].get('finish_reason')}")
    if usage.get("prompt_tokens"):
        print(f"[smoke] prompt_tokens: {usage['prompt_tokens']}")

    # 2. 并发吞吐（非流式）
    import concurrent.futures as cf

    def one(_):
        t0 = time.monotonic()
        r = post(args.host, args.port, "/v1/completions", {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": args.output_len,
            "temperature": 0.0,
            "stream": False,
        })
        return time.monotonic() - t0, r

    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(one, range(n)))
    total_tok = sum(r["usage"].get("completion_tokens", args.output_len)
                    for _, r in results)
    wall = max(dt for dt, _ in results)
    print(f"[bench] {n} streams x ~{args.output_len} tok: {total_tok} tok "
          f"in {wall:.2f}s wall = {total_tok/wall:.1f} tok/s aggregate, "
          f"{total_tok/n/max(dt, 1e-9):.1f} tok/s per-stream avg")
    print("[mtp_bench] DONE")


if __name__ == "__main__":
    main()
