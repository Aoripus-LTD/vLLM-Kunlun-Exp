#!/usr/bin/env python
"""放宽 Mamba speculative 白名单，纳入 qwen3_5。

vllm/model_executor/layers/mamba/abstract.py 的 get_kv_cache_spec 在
speculative 模式下对非 qwen3_next 的 model_type 直接 raise
NotImplementedError。Qwen3.8 的 model_type 是 qwen3_5（vllm-kunlun 型号），
其 GDN 层 Qwen3_5GatedDeltaNet 继承 Qwen3NextGatedDeltaNet(nn.Module,
MambaBase)，speculative 路径（num_spec state 预留）已就绪，只差白名单豁免。

Usage: python fix_mamba_spec.py --apply <abstract.py>
       python fix_mamba_spec.py --test <site-packages-root>
"""
import sys

OLD = 'not in ["qwen3_next"]'

NEW = 'not in ["qwen3_next", "qwen3_5", "qwen3_5_text"]'


def apply(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    n = src.count(OLD)
    assert n == 1, f"old block occurrences = {n} (expected 1)"
    src = src.replace(OLD, NEW)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[apply] patched {path} OK")


def test(site_packages: str) -> None:
    p = f"{site_packages}/vllm/model_executor/layers/mamba/abstract.py"
    src = open(p, encoding="utf-8").read()
    assert 'not in ["qwen3_next", "qwen3_5", "qwen3_5_text"]' in src, \
        "widened whitelist not found in abstract.py"
    assert src.count('not in ["qwen3_next", "qwen3_5", "qwen3_5_text"]') == 1
    print("[test] PASS  whitelist widened to include qwen3_5/qwen3_5_text")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
