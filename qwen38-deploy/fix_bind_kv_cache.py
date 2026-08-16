#!/usr/bin/env python
"""bind_kv_cache 放宽平台判定，纳入 OOT（昆仑芯）平台。

vllm/v1/worker/utils.py 的 bind_kv_cache 在 `len(layer_names) > 1` 时
（同一 layer_index 有多个 kv cache 层，典型如 MTP drafter 的
mtp.layers.0.self_attn.attn 与 target 的
language_model.model.layers.0.linear_attn 同为 index 0）依赖
is_cuda_alike()/is_xpu()/is_cpu() 判定放行。昆仑芯是 CUDA 兼容的 OOT
平台，is_cuda_alike() 返回 False（_enum=OOT 不在 CUDA/ROCM 集合），
导致 raise NotImplementedError 而无法启动 MTP。

修复：条件追加 is_out_of_tree()，让昆仑芯与 CUDA 一样走 pass 分支。

Usage: python fix_bind_kv_cache.py --apply <utils.py>
       python fix_bind_kv_cache.py --test <site-packages-root>
"""
import sys

OLD_IF = """            if (
                current_platform.is_cuda_alike()
                or current_platform.is_xpu()
                or current_platform.is_cpu()
            ):"""

NEW_IF = """            if (
                current_platform.is_cuda_alike()
                or current_platform.is_xpu()
                or current_platform.is_cpu()
                or current_platform.is_out_of_tree()
            ):"""

# 诊断期间临时改成带 layer_index/layer_names 的 raise，恢复为原始。
OLD_RAISE = """                raise NotImplementedError(
                    "multiple kv cache layers layer_index=" + str(layer_index) + " layers=" + str(layer_names)
                )"""

NEW_RAISE = """                raise NotImplementedError"""


def apply(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    n_if = src.count(OLD_IF)
    assert n_if == 1, f"old if occurrences = {n_if} (expected 1)"
    src = src.replace(OLD_IF, NEW_IF)
    # 诊断 raise 若存在则恢复；不存在则跳过（幂等）。
    if OLD_RAISE in src:
        src = src.replace(OLD_RAISE, NEW_RAISE)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[apply] patched {path} OK")


def test(site_packages: str) -> None:
    p = f"{site_packages}/vllm/v1/worker/utils.py"
    src = open(p, encoding="utf-8").read()
    assert "current_platform.is_out_of_tree()" in src, \
        "is_out_of_tree() not found in bind_kv_cache"
    assert 'raise NotImplementedError("multiple kv cache layers' not in src, \
        "diagnostic raise not reverted"
    print("[test] PASS  bind_kv_cache allows OOT platform, diag raise reverted")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
