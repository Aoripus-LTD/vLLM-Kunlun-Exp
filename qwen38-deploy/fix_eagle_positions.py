#!/usr/bin/env python
"""把 vllm-kunlun eagle.py 的 self.positions 直接访问改成 _set/_get_positions。

vllm 0.15.1 的 EagleProposer 只在 uses_mrope=False 时定义 self.positions，
uses_mrope=True 时定义 self.mrope_positions。vllm-kunlun 的 propose/
prepare_inputs 函数体直接访问 self.positions，导致 mrope 模型（Qwen3.8
多模态 wrapper）运行时 `'EagleProposer' object has no attribute 'positions'`。

修复：把 4 处 self.positions 访问改成语义正确的 _set_positions/_get_positions
方法（vllm 本体已有的 mrope 兼容方法）。

Usage: python fix_eagle_positions.py --apply <eagle.py>
       python fix_eagle_positions.py --test <site-packages-root>
"""
import sys

REPLACEMENTS = [
    (
        "    self.positions[:num_tokens] = target_positions",
        "    self._set_positions(num_tokens, target_positions)",
    ),
    (
        "            positions=self.positions[:num_input_tokens],",
        "            positions=self._get_positions(num_input_tokens),",
    ),
    (
        "        self.positions[:batch_size] = clamped_positions",
        "        self._set_positions(batch_size, clamped_positions)",
    ),
    (
        "                positions=self.positions[:input_batch_size],",
        "                positions=self._get_positions(input_batch_size),",
    ),
]


def apply(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    for old, new in REPLACEMENTS:
        n = src.count(old)
        assert n == 1, f"pattern occurrences = {n} (expected 1): {old!r}"
        src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[apply] patched {path} OK")


def test(site_packages: str) -> None:
    p = f"{site_packages}/vllm_kunlun/v1/sample/spec_decode/eagle.py"
    src = open(p, encoding="utf-8").read()
    assert "self._set_positions(num_tokens, target_positions)" in src
    assert "self._get_positions(num_input_tokens)" in src
    assert "self._set_positions(batch_size, clamped_positions)" in src
    assert "self._get_positions(input_batch_size)" in src
    assert "self.positions[" not in src, "direct self.positions access still present"
    print("[test] PASS  self.positions replaced with _set/_get_positions")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
