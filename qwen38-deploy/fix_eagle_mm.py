#!/usr/bin/env python
"""Fall back to image_token_id when the target config lacks image_token_index.

vllm/v1/spec_decode/eagle.py syncs the drafter's image_token_index from the
target config when the target reports multimodal support. The Qwen3_5Config
wrapper carries image_token_id but no image_token_index, so the sync raises
AttributeError for the (text-only) Qwen3.8 model. Fall back to
image_token_id, which is the value the Qwen2_5_VL/Qwen3VL branches use.

Usage: python fix_eagle_mm.py --apply <eagle.py>
       python fix_eagle_mm.py --test <site-packages-root>
"""
import sys

OLD = """            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
"""

NEW = """            else:
                self.model.config.image_token_index = (
                    getattr(target_model.config, "image_token_index",
                            target_model.config.image_token_id)
                )
"""


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
    p = f"{site_packages}/vllm/v1/spec_decode/eagle.py"
    src = open(p, encoding="utf-8").read()
    assert 'getattr(target_model.config, "image_token_index",' in src, \
        "getattr fallback not found in eagle.py"
    assert 'target_model.config.image_token_id)' in src
    print("[test] PASS  getattr fallback present")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
