#!/usr/bin/env python3
# 05_prefix_bench.py — 前缀缓存命中验证 + 短吞吐采样（vllm serve OpenAI API）
# 流程: 1) 共享前缀 ~8K+ tokens × 32 请求, 冷缓存一轮 → 热缓存一轮, 对比 TTFT/总耗时
#       2) /metrics 抓 prefix/cache 指标, 实测命中率
#       3) 64 并发 512/512 短吞吐采样（C 阶段用, 配合 engine 日志做 step 分解）
# 用法: /opt/vllm_kunlun/bin/python 05_prefix_bench.py   （容器内, 连 127.0.0.1:8000）
import http.client
import json
import statistics
import threading
import time

HOST = "127.0.0.1"
PORT = 8000
MODEL = None

# 共享前缀: 固定科技文本重复 ~13K 字符 ≈ 8K+ tokens
UNIT = ("随着大规模语言模型的规模化训练持续推进，其在长文本理解、代码生成与多轮"
        "对话等场景下的能力边界不断扩展。模型架构从标准注意力机制逐步演进到混合"
        "架构：部分层采用线性注意力以降低长序列的计算复杂度，其余层保留全注意力"
        "以维持信息检索精度。量化技术在保持推理精度的前提下显著压缩显存占用，"
        "W8A8 格式将权重与激活统一量化为 8 位整数，通过反量化恢复计算域精度。"
        "服务端推理框架通过连续批处理、分块预填充与缓存复用等手段提升吞吐，"
        "其中前缀缓存对共享上下文的请求可显著降低首 token 延迟。")
PREFIX = UNIT * 20  # ~13K 字符


def api(path, payload=None, method="GET", timeout=900):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    body = json.dumps(payload) if payload else None
    conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read().decode("utf-8", "replace")
    conn.close()
    return data


def get_model():
    m = json.loads(api("/v1/models"))
    return m["data"][0]["id"]


def ask_stream(prompt, max_tokens=64):
    """流式单请求, 返回 (TTFT, 总耗时)"""
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "stream": True}
    conn = http.client.HTTPConnection(HOST, PORT, timeout=1200)
    conn.request("POST", "/v1/completions",
                 body=json.dumps(payload), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    t0 = time.time()
    ttft = None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:"):
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                j = json.loads(d)
            except Exception:
                continue
            if ttft is None and j.get("choices") and j["choices"][0].get("text"):
                ttft = time.time() - t0
    dt = time.time() - t0
    conn.close()
    return ttft, dt


def run_round(name, n=32):
    prompts = [PREFIX + f"\n问题: 请用一句话回答第 {i} 点的核心内容。" for i in range(n)]
    ttft, dt = [None] * n, [None] * n

    def f(i):
        ttft[i], dt[i] = ask_stream(prompts[i])

    ths = [threading.Thread(target=f, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    ok = [i for i in range(n) if dt[i] is not None]
    print(f"[{name}] 完成 {len(ok)}/{n}  总耗时 {wall:.1f}s")
    t2 = [ttft[i] for i in ok if ttft[i] is not None]
    if t2:
        print(f"  TTFT: mean {statistics.mean(t2):.2f}s  min {min(t2):.2f}s  max {max(t2):.2f}s")
    return wall


def metrics_cache():
    m = api("/metrics")
    print("[metrics] prefix/cache/kv 相关指标:")
    for line in m.splitlines():
        if any(k in line for k in ("prefix", "cache", "kv")) and not line.startswith("#"):
            print("  ", line[:180])


def throughput_sample(n=64, out_len=512):
    """64 并发短吞吐: 不同 prompt, 统计墙钟与 metrics 差值"""
    base = api("/metrics")
    g0 = _metric(base, "vllm:generation_tokens_total")
    p0 = _metric(base, "vllm:prompt_tokens_total")
    prompts = [f"请用中文写一段关于主题{i}的技术说明。" + ("这是一段填充文本，用于控制输入长度。" * 30)
               for i in range(n)]
    ttft, dt = [None] * n, [None] * n

    def f(i):
        ttft[i], dt[i] = ask_stream(prompts[i], max_tokens=out_len)

    ths = [threading.Thread(target=f, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    m2 = api("/metrics")
    g1 = _metric(m2, "vllm:generation_tokens_total")
    p1 = _metric(m2, "vllm:prompt_tokens_total")
    ok = [i for i in range(n) if dt[i] is not None]
    print(f"[throughput] {len(ok)}/{n} 完成  墙钟 {wall:.1f}s")
    print(f"  prompt tokens 增量: {p1 - p0}  generation tokens 增量: {g1 - g0}")
    if wall > 0 and (g1 - g0) > 0:
        print(f"  生成吞吐: {(g1 - g0) / wall:.1f} output tok/s（8 卡合计）")


def _metric(metrics_text, name):
    # vllm 0.15.1 指标行带 label: `vllm:generation_tokens_total{engine="0",...} 123`
    for line in metrics_text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            return int(float(line.split()[-1]))
    return 0


if __name__ == "__main__":
    MODEL = get_model()
    print(f"model = {MODEL}")
    print("== 第一轮: 冷缓存（32 并发共享前缀）==")
    w1 = run_round("cold", 32)
    time.sleep(3)
    metrics_cache()
    print("== 第二轮: 热缓存（同 32 请求再发）==")
    w2 = run_round("hot", 32)
    print(f"[对比] cold {w1:.1f}s -> hot {w2:.1f}s, 加速 {(w1 / w2):.1f}x")
    metrics_cache()
    print("== 吞吐采样 ==")
    throughput_sample()
