#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""注册 Qwen3_5MTP（PR 423 移植）到模型 registry，并让 MTP draft 架构映射指向它。

改动 1（--apply-models）：vllm_kunlun/models/__init__.py 注册
  Qwen3_5MTP / Qwen3_5MoeMTP（vllm_kunlun.models.qwen3_5_mtp）。

改动 2（--apply-spec）：vllm/config/speculative.py 的 qwen3_5 draft 分支
  由 Qwen3NextMTP 改为 Qwen3_5MTP（PR 423 的 Qwen3.5 专属 MTP head）。

Usage: python fix_mtp_registry.py --apply-models <file>
       python fix_mtp_registry.py --apply-spec <file>
       python fix_mtp_registry.py --test <site-packages-root>
"""
import sys

MODELS_ANCHOR = (
    '    ModelRegistry.register_model(\n'
    '        "Qwen3_5ForConditionalGeneration",\n'
    '        "vllm_kunlun.models.qwen3_5:Qwen3_5ForConditionalGeneration",\n'
    '    )'
)

MODELS_ADD = MODELS_ANCHOR + (
    '\n'
    '\n'
    '    ModelRegistry.register_model(\n'
    '        "Qwen3_5MTP",\n'
    '        "vllm_kunlun.models.qwen3_5_mtp:Qwen3_5MTP",\n'
    '    )\n'
    '\n'
    '    ModelRegistry.register_model(\n'
    '        "Qwen3_5MoeMTP",\n'
    '        "vllm_kunlun.models.qwen3_5_mtp:Qwen3_5MoeMTP",\n'
    '    )'
)

SPEC_OLD = (
    '            hf_config.model_type = "qwen3_next"\n'
    '        if hf_config.model_type == "qwen3_next":\n'
    '            hf_config.model_type = "qwen3_next_mtp"\n'
    '        if hf_config.model_type == "qwen3_next_mtp":\n'
    '            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)\n'
    '            hf_config.update(\n'
    '                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}\n'
    '            )'
)

SPEC_NEW = (
    '            hf_config.model_type = "qwen3_5_mtp"\n'
    '            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)\n'
    '            hf_config.update(\n'
    '                {"n_predict": n_predict, "architectures": ["Qwen3_5MTP"]}\n'
    '            )\n'
    '        if hf_config.model_type == "qwen3_next":\n'
    '            hf_config.model_type = "qwen3_next_mtp"\n'
    '        if hf_config.model_type == "qwen3_next_mtp":\n'
    '            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)\n'
    '            hf_config.update(\n'
    '                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}\n'
    '            )'
)


def _apply(path: str, old: str, new: str, what: str) -> None:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if new in src:
        print(f"[{what}] already applied, skip")
        return
    n = src.count(old)
    assert n == 1, f"[{what}] anchor occurrences = {n} (expected 1)"
    src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[{what}] patched {path} OK")


def apply_models(path: str) -> None:
    _apply(path, MODELS_ANCHOR, MODELS_ADD, "models")


def apply_spec(path: str) -> None:
    _apply(path, SPEC_OLD, SPEC_NEW, "spec")


def test(site_packages: str) -> None:
    m = open(
        f"{site_packages}/vllm_kunlun/models/__init__.py", encoding="utf-8"
    ).read()
    assert '"Qwen3_5MTP"' in m and '"Qwen3_5MoeMTP"' in m, "models registry missing"
    s = open(f"{site_packages}/vllm/config/speculative.py", encoding="utf-8").read()
    assert '"architectures": ["Qwen3_5MTP"]' in s, "spec draft mapping missing"
    assert '"architectures": ["Qwen3NextMTP"]' in s, "qwen3_next branch removed!"
    print("[test] PASS  Qwen3_5MTP registered, qwen3_next branch intact")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply-models":
        apply_models(sys.argv[2])
    elif mode == "--apply-spec":
        apply_spec(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
