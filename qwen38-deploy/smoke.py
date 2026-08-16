#!/usr/bin/env python3
# smoke.py — 单请求冒烟：8K 共享前缀 + 流式响应检查（不吞错误）
import http.client
import json
import sys
import time

UNIT = ("随着大规模语言模型的规模化训练持续推进，其在长文本理解、代码生成与多轮"
        "对话等场景下的能力边界不断扩展。模型架构从标准注意力机制逐步演进到混合"
        "架构：部分层采用线性注意力以降低长序列的计算复杂度，其余层保留全注意力"
        "以维持信息检索精度。量化技术在保持推理精度的前提下显著压缩显存占用，"
        "W8A8 格式将权重与激活统一量化为 8 位整数，通过反量化恢复计算域精度。"
        "服务端推理框架通过连续批处理、分块预填充与缓存复用等手段提升吞吐，"
        "其中前缀缓存对共享上下文的请求可显著降低首 token 延迟。")
PREFIX = UNIT * 20  # ~8K tokens

resp = None
try:
    conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=600)
    payload = {"model": "/home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic",
               "prompt": PREFIX + "\n问题: 请用一句话回答。",
               "max_tokens": 32, "stream": True}
    conn.request("POST", "/v1/completions", body=json.dumps(payload),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    print("HTTP", resp.status, resp.reason)
    t0 = time.time()
    n = 0
    ttft = None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            break
        try:
            j = json.loads(d)
        except Exception as e:
            print("parse err:", e, line[:200])
            continue
        if j.get("error"):
            print("STREAM ERROR:", json.dumps(j["error"])[:500])
            sys.exit(1)
        if j.get("choices") and j["choices"][0].get("text"):
            n += 1
            if ttft is None:
                ttft = time.time() - t0
    dt = time.time() - t0
    conn.close()
    print(f"tokens={n} ttft={ttft:.2f}s total={dt:.2f}s")
    if n == 0:
        print("NO TOKENS GENERATED")
        sys.exit(2)
    print("SMOKE OK")
except Exception as e:
    print("EXCEPTION:", type(e).__name__, e)
    sys.exit(3)
