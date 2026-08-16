#!/usr/bin/env python
"""把 spec 路径的 num_accepted_tokens 截断到 num_spec_decodes。

昆仑芯 fused_recurrent_gated_delta_rule_fwdv2 断言 num_accepted_tokens
大小为 N（= len(cu_seqlens) - 1 = num_spec_decodes）。gdn_attn.py 在 CUDA
graph capture 阶段把 num_accepted_tokens 扩展到 batch_size
（= m.num_actual_tokens，含 padding），导致大小超过 N。修复：调用时截断到
前 num_spec_decodes 个。

Usage: python fix_num_accepted.py --apply <qwen3_next.py>
       python fix_num_accepted.py --test <site-packages-root>
"""
import sys

OLD = """                ssm_state_indices=spec_state_indices_tensor[:, 0],
                num_accepted_tokens=num_accepted_tokens,"""

NEW = """                ssm_state_indices=spec_state_indices_tensor[:, 0],
                num_accepted_tokens=num_accepted_tokens[: attn_metadata.num_spec_decodes],"""


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
    assert "num_accepted_tokens=num_accepted_tokens[: attn_metadata.num_spec_decodes]," in src, \
        "num_accepted_tokens slice not found in qwen3_next.py"
    print("[test] PASS  num_accepted_tokens truncated to num_spec_decodes")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
