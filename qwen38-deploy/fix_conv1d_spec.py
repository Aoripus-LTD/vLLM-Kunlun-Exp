#!/usr/bin/env python
"""修复 GDN 层 speculative 模式 conv_state 更新硬编码 kernel 宽度。

vllm_kunlun/ops/mamba/causal_conv1d.py 的 torch_causal_conv1d_update_spec
（speculative decode 用）硬编码 `tmp_conv_state[: (2 + num_accepted_tokens)]`，
其中 `2` 假设 conv kernel 宽度为 3（state_len = kernel-1 = 2）。Qwen3.8 的
GDN 层 `linear_conv_kernel_dim = 4`，其 conv_state 的 state 维度为
`kernel_size - 1 + num_spec = 4`，导致 slice 只取到 2 个位置而赋值目标
需要 4 个位置，报 `RuntimeError: expanded size (4) must match (2)`。

修复：用 `weight.shape[1] - 1`（= kernel 宽度 - 1）替换硬编码的 2。

Usage: python fix_conv1d_spec.py --apply <causal_conv1d.py>
       python fix_conv1d_spec.py --test <site-packages-root>
"""
import sys

OLD = "            [tmp_conv_state[: (2 + num_accepted_tokens[i]), :], hidden_states_i], dim=0"

NEW = ("            [tmp_conv_state[: (weight.shape[1] - 1 + int(num_accepted_tokens[i])), :], "
       "hidden_states_i], dim=0")


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
    p = f"{site_packages}/vllm_kunlun/ops/mamba/causal_conv1d.py"
    src = open(p, encoding="utf-8").read()
    assert "weight.shape[1] - 1 + int(num_accepted_tokens[i])" in src, \
        "kernel-width fix not found in causal_conv1d.py"
    assert src.count("weight.shape[1] - 1 + int(num_accepted_tokens[i])") == 1
    print("[test] PASS  conv_state slice uses weight width and int() cast")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
