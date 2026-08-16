#!/usr/bin/env python
"""给 EagleProposer 补 use_cuda_graph / cudagraph_batch_sizes 默认值。

vllm-kunlun 从新版本移植的 propose 函数体访问 self.use_cuda_graph 与
self.cudagraph_batch_sizes，但 vllm 0.15.1 的 EagleProposer 没有这两个
属性，导致运行时 `'EagleProposer' object has no attribute 'use_cuda_graph'`。

修复：monkeypatch 前补默认值 use_cuda_graph=False、cudagraph_batch_sizes=[]，
让 propose 走非 cudagraph 分支（gpu_model_runner 已在外部完成 padding）。

Usage: python fix_eagle_cudagraph.py --apply <eagle.py>
       python fix_eagle_cudagraph.py --test <site-packages-root>
"""
import sys

OLD = """EagleProposer.propose = propose
EagleProposer.prepare_next_token_ids_padded = prepare_next_token_ids_padded"""

NEW = """EagleProposer.use_cuda_graph = False
EagleProposer.cudagraph_batch_sizes = []
EagleProposer.propose = propose
EagleProposer.prepare_next_token_ids_padded = prepare_next_token_ids_padded"""


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
    p = f"{site_packages}/vllm_kunlun/v1/sample/spec_decode/eagle.py"
    src = open(p, encoding="utf-8").read()
    assert "EagleProposer.use_cuda_graph = False" in src, \
        "use_cuda_graph default not found"
    assert "EagleProposer.cudagraph_batch_sizes = []" in src, \
        "cudagraph_batch_sizes default not found"
    print("[test] PASS  use_cuda_graph / cudagraph_batch_sizes defaults added")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
