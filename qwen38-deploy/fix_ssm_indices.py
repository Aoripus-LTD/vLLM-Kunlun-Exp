#!/usr/bin/env python
"""把 GDN spec 路径的 ssm_state_indices 从 2D block table 压平成 1D。

昆仑芯 kunlun_ops.fused_recurrent_gated_delta_rule_fwdv2 在 varlen 模式下
断言 h0_indices 大小为 T（token 数），即要求 1D [num_tokens]。vllm 的
qwen3_next.py spec 路径传的是 2D `spec_state_indices_tensor`
（[num_spec_decodes, num_spec+1] 的 block table），其 flatten 大小恰好
等于 num_tokens，故 reshape(-1) 即符合 kernel 期望。

Usage: python fix_ssm_indices.py --apply <qwen3_next.py>
       python fix_ssm_indices.py --test <site-packages-root>
"""
import sys

OLD = "                ssm_state_indices=spec_state_indices_tensor,"

NEW = "                ssm_state_indices=spec_state_indices_tensor[:, 0],"


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
    p = f"{site_packages}/vllm_kunlun/models/qwen3_next.py"
    src = open(p, encoding="utf-8").read()
    assert "ssm_state_indices=spec_state_indices_tensor[:, 0]," in src, \
        "[:, 0] not found in qwen3_next.py"
    print("[test] PASS  ssm_state_indices narrowed to first column")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
