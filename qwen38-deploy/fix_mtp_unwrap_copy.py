#!/usr/bin/env python
"""Qwen3NextMTP 解包前深拷贝 VllmConfig，避免污染 target model。

Qwen3NextMTP.__init__（vllm/model_executor/models/qwen3_next_mtp.py）在
解包 text_config 时 in-place 执行 `vllm_config.model_config.hf_config =
text_cfg`。当 drafter 与 target 共享同一个 VllmConfig 时，这个 in-place
替换把 target 的 hf_config 从 Qwen3_5Config（model_type=qwen3_5，
architectures 有值）替换成 Qwen3_5TextConfig（model_type=qwen3_5_text，
architectures=None），导致后续 target 的 mamba 白名单匹配失败
（NotImplementedError）和 get_model_architecture 报
`TypeError: 'NoneType' object is not iterable`。

修复：函数入口 deepcopy VllmConfig，之后的所有 in-place 修改只作用于
副本，target 的 VllmConfig 保持原样。

Usage: python fix_mtp_unwrap_copy.py --apply <qwen3_next_mtp.py>
       python fix_mtp_unwrap_copy.py --test <site-packages-root>
"""
import sys

OLD = """        config = vllm_config.model_config.hf_config
        # Same unwrap as in Qwen3NextMultiTokenPredictor.
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None and not hasattr(config, "vocab_size"):
            config = text_cfg
            # Rewrite the shared config so every downstream reader
            # (decoder layers, attention) sees the unwrapped config.
            vllm_config.model_config.hf_config = config
"""

NEW = """        import copy
        # Deep-copy the VllmConfig so the drafter's config unwrap below
        # never mutates the target model's shared config. The target must
        # keep its own hf_config (model_type qwen3_5, architectures
        # populated) for the mamba speculative whitelist and profile_run.
        vllm_config = copy.deepcopy(vllm_config)
        config = vllm_config.model_config.hf_config
        # Same unwrap as in Qwen3NextMultiTokenPredictor.
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None and not hasattr(config, "vocab_size"):
            config = text_cfg
            # Rewrite the drafter's copied config so every downstream reader
            # (decoder layers, attention) sees the unwrapped config.
            vllm_config.model_config.hf_config = config
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
    p = f"{site_packages}/vllm/model_executor/models/qwen3_next_mtp.py"
    src = open(p, encoding="utf-8").read()
    assert "vllm_config = copy.deepcopy(vllm_config)" in src, \
        "deepcopy line not found in qwen3_next_mtp.py"
    assert src.count("vllm_config = copy.deepcopy(vllm_config)") == 1
    print("[test] PASS  Qwen3NextMTP deep-copies VllmConfig before unwrap")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
