#!/usr/bin/env python
"""hf_config_override 深拷贝，避免污染 target model 的 config。

SpeculativeConfig.hf_config_override（vllm/config/speculative.py）作为
drafter 的 hf_overrides 回调，in-place 把 hf_config.model_type 从 qwen3_5
改成 qwen3_next_mtp。当 target 与 drafter 共享同一 config 对象（相同 model
路径 + config 加载缓存/共享引用）时，in-place 修改会污染 target 的
model_type，导致 mamba/abstract.py 的 qwen3_5 白名单匹配失败
（NotImplementedError: Mamba with speculative decoding is not supported yet.）。

修复：函数入口 deepcopy，只改副本并返回副本，target 的 config 保持
model_type == "qwen3_5"。

Usage: python fix_hf_override_deepcopy.py --apply <speculative.py>
       python fix_hf_override_deepcopy.py --test <site-packages-root>
"""
import sys

OLD = """    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
"""

NEW = """    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        import copy
        # Deep-copy so the drafter's model_type rewrite (qwen3_5 ->
        # qwen3_next_mtp) never mutates the target model's shared config.
        # The target must keep model_type == "qwen3_5" for the mamba
        # speculative whitelist in model_executor/layers/mamba/abstract.py.
        hf_config = copy.deepcopy(hf_config)
        initial_architecture = hf_config.architectures[0]
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
    p = f"{site_packages}/vllm/config/speculative.py"
    src = open(p, encoding="utf-8").read()
    assert "hf_config = copy.deepcopy(hf_config)" in src, \
        "deepcopy line not found in speculative.py"
    assert src.count("hf_config = copy.deepcopy(hf_config)") == 1
    print("[test] PASS  hf_config_override deep-copies before mutating")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
