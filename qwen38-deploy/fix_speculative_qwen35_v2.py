#!/usr/bin/env python
"""v2 fix for Qwen3.5 MTP: promote text sub-config fields in place.

The hf_config_override hook is called by ModelConfig as an in-place mutator
(side-effect only; its return value is discarded by the caller). The v1 patch
built a fresh Qwen3NextConfig and rebound the local name, so the promoted
fields (vocab_size etc.) never landed on the wrapper object that the drafter
later reads. v2 writes the fields onto the existing Qwen3_5Config in place.

Usage: python fix_speculative_qwen35_v2.py --apply <speculative.py>
       python fix_speculative_qwen35_v2.py --test <site-packages-root>
"""
import sys

OLD = """            from vllm.transformers_utils.configs import Qwen3NextConfig
            text_cfg = getattr(hf_config, "text_config", None)
            if text_cfg is not None:
                rebuilt = Qwen3NextConfig()
                for k, v in text_cfg.to_dict().items():
                    if hasattr(rebuilt, k) and k != "model_type":
                        setattr(rebuilt, k, v)
                for k in ("max_position_embeddings", "num_nextn_predict_layers"):
                    if getattr(hf_config, k, None) is not None:
                        setattr(rebuilt, k, getattr(hf_config, k))
                hf_config = rebuilt
            hf_config.model_type = "qwen3_next\""""

NEW = """            text_cfg = getattr(hf_config, "text_config", None)
            if text_cfg is not None:
                # In-place mutation: hf_overrides is a side-effect hook whose
                # return value is discarded, so promote the text sub-config
                # fields onto the wrapper object instead of rebuilding a new
                # config instance.
                for k, v in text_cfg.to_dict().items():
                    if not hasattr(hf_config, k):
                        setattr(hf_config, k, v)
            if not hasattr(hf_config, "num_nextn_predict_layers"):
                # Qwen3.8 configs carry no num_nextn_predict_layers; the MTP
                # head defaults to a single layer, so make it explicit.
                hf_config.num_nextn_predict_layers = 1
            hf_config.model_type = "qwen3_next\""""

MODEL = "/home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic"


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
    sys.path.insert(0, site_packages)
    from transformers import AutoConfig  # noqa: E402

    from vllm.config.speculative import SpeculativeConfig  # noqa: E402

    c = AutoConfig.from_pretrained(MODEL, trust_remote_code=False)
    out = SpeculativeConfig.hf_config_override(c)

    checks = [
        ("same object (in-place)", out is c),
        ("type stays Qwen3_5Config", type(out).__name__ == "Qwen3_5Config"),
        ("model_type == qwen3_next_mtp", out.model_type == "qwen3_next_mtp"),
        ("architectures == [Qwen3NextMTP]",
         getattr(out, "architectures", None) == ["Qwen3NextMTP"]),
        ("vocab_size == 248320", getattr(out, "vocab_size", None) == 248320),
        ("hidden_size == 5120", getattr(out, "hidden_size", None) == 5120),
        ("num_hidden_layers == 64",
         getattr(out, "num_hidden_layers", None) == 64),
        ("rms_norm_eps present",
         getattr(out, "rms_norm_eps", None) is not None),
        ("n_predict set", getattr(out, "n_predict", None) is not None),
        ("text_config intact",
         out.text_config.model_type == "qwen3_5_text"),
    ]
    ok = True
    for name, cond in checks:
        print(f"[test] {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and cond
    assert ok, "unit test failed"
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
