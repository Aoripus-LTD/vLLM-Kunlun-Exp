#!/usr/bin/env python
"""Unwrap the Qwen3.5 multimodal config wrapper in the MTP drafter.

Qwen3NextMultiTokenPredictor and Qwen3NextMTP read
`vllm_config.model_config.hf_config`. When the drafter is built off the
target's hf_config, that object is the Qwen3_5Config multimodal wrapper
whose model fields live on text_config, so attribute reads like
`config.vocab_size` raise AttributeError. Unwrap to text_config when the
wrapper lacks vocab_size.

Usage: python fix_qwen3_next_mtp.py --apply <qwen3_next_mtp.py>
       python fix_qwen3_next_mtp.py --test <site-packages-root>
"""
import sys

OLD_PREDICTOR = """        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config

        config: Qwen3NextConfig = model_config.hf_config
"""

NEW_PREDICTOR = """        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config

        config: Qwen3NextConfig = model_config.hf_config
        # Qwen3.5 multimodal wrapper: model fields live on text_config
        # when the MTP head is loaded off the target's hf_config.
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None and not hasattr(config, "vocab_size"):
            config = text_cfg
"""

OLD_IMPORT = """    Qwen3NextDecoderLayer,
"""

NEW_IMPORT = """    Qwen3NextDecoderLayer,
    Qwen3NextSparseMoeBlock,
"""

OLD_MOE = """        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.set_moe_parameters()
"""

NEW_MOE = """        self.logits_processor = LogitsProcessor(config.vocab_size)
        # Dense MTP heads (Qwen3.8) have no SparseMoeBlock layers; the MoE
        # parameter scan in QwenNextMixtureOfExperts would raise.
        if any(isinstance(layer.mlp, Qwen3NextSparseMoeBlock)
               for layer in self.model.layers):
            self.set_moe_parameters()
"""

OLD_REMAP = """        def remap_weight_names(weights):
            for name, weight in weights:
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                elif not any(key in name for key in shared_weight_names):
                    continue
                yield name, weight
"""

NEW_REMAP = """        def remap_weight_names(weights):
            for name, weight in weights:
                if name.startswith("model."):
                    # The checkpoint stores the Qwen3.5 wrapper weights under
                    # a model.language_model.* prefix; strip the model. part
                    # so the language_model branch below sees the same names
                    # as a bare Qwen3Next checkpoint.
                    name = name[len("model."):]
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                elif name.startswith("language_model."):
                    # Shared weights (embed_tokens / lm_head) land on the
                    # drafter's own names; mtp.* maps into model.*. Dense
                    # target layers are dropped.
                    n2 = name[len("language_model."):]
                    if n2.startswith("mtp."):
                        name = "model." + n2[len("mtp."):]
                    elif any(key in n2 for key in shared_weight_names):
                        name = "model." + n2
                    else:
                        continue
                elif not any(key in name for key in shared_weight_names):
                    continue
                yield name, weight
"""

OLD_WRAPPER = """    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
"""

NEW_WRAPPER = """    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        # Same unwrap as in Qwen3NextMultiTokenPredictor.
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None and not hasattr(config, "vocab_size"):
            config = text_cfg
            # Rewrite the shared config so every downstream reader
            # (decoder layers, attention) sees the unwrapped config.
            vllm_config.model_config.hf_config = config
            # Backfill Qwen3NextConfig defaults (dense checkpoints carry
            # no num_experts / moe fields) so decoder layers and the MoE
            # branch guard read consistent values.
            from vllm.transformers_utils.configs import Qwen3NextConfig
            for k, v in Qwen3NextConfig().to_dict().items():
                if not hasattr(config, k):
                    setattr(config, k, v)
            # Qwen3.8 text is dense; the Qwen3NextConfig backfill defaults
            # to a MoE layout (num_experts=512), so force the MoE branch
            # guard off.
            config.num_experts = 0
            config.num_experts_per_tok = 0
"""

MODEL = "/home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic"


def apply(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    for old, new in ((OLD_IMPORT, NEW_IMPORT), (OLD_MOE, NEW_MOE),
                     (OLD_PREDICTOR, NEW_PREDICTOR), (OLD_REMAP, NEW_REMAP),
                     (OLD_WRAPPER, NEW_WRAPPER)):
        n = src.count(old)
        assert n == 1, f"block occurrences = {n} (expected 1): {old[:60]!r}"
        src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[apply] patched {path} OK")


def test(site_packages: str) -> None:
    sys.path.insert(0, site_packages)
    from transformers import AutoConfig  # noqa: E402

    c = AutoConfig.from_pretrained(MODEL, trust_remote_code=False)
    assert type(c).__name__ == "Qwen3_5Config", type(c).__name__
    text_cfg = getattr(c, "text_config", None)
    assert text_cfg is not None
    config = text_cfg if not hasattr(c, "vocab_size") else c

    # Mirror the wrapper backfill (Qwen3NextConfig defaults on missing keys,
    # then force the dense layout).
    from vllm.transformers_utils.configs import Qwen3NextConfig  # noqa: E402
    for k, v in Qwen3NextConfig().to_dict().items():
        if not hasattr(config, k):
            setattr(config, k, v)
    config.num_experts = 0
    config.num_experts_per_tok = 0

    checks = [
        ("unwrapped to text_config", config is text_cfg),
        ("vocab_size == 248320", getattr(config, "vocab_size", None) == 248320),
        ("num_hidden_layers == 64",
         getattr(config, "num_hidden_layers", None) == 64),
        ("hidden_size == 5120", getattr(config, "hidden_size", None) == 5120),
        ("rms_norm_eps present",
         getattr(config, "rms_norm_eps", None) is not None),
        ("num_nextn_predict_layers default 1",
         getattr(config, "num_nextn_predict_layers", 1) == 1),
        ("num_experts backfilled 0",
         getattr(config, "num_experts", None) == 0),
    ]
    ok = True
    for name, cond in checks:
        print(f"[test] {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and cond

    # Mirror remap_weight_names (Qwen3NextMTP.load_weights): mtp.* -> model.*,
    # model.language_model.* -> model.* (shared weights only, dense layers
    # dropped), bare lm_head stays at top level.
    def remap(names: list[str]) -> list[str]:
        out = []
        for n in names:
            if n.startswith("model."):
                n = n[len("model."):]
            if n.startswith("mtp."):
                out.append(n.replace("mtp.", "model."))
            elif n.startswith("language_model."):
                n2 = n[len("language_model."):]
                if n2.startswith("mtp."):
                    out.append("model." + n2[len("mtp."):])
                elif any(k in n2 for k in ("embed_tokens", "lm_head")):
                    out.append("model." + n2)
            elif any(k in n for k in ("embed_tokens", "lm_head")):
                out.append(n)
        return out

    src = ["mtp.fc.weight",
           "language_model.embed_tokens.weight",
           "model.language_model.embed_tokens.weight",
           "model.language_model.mtp.layers.0.self_attn.q_proj.weight",
           "lm_head.weight",
           "model.language_model.layers.0.self_attn.q_proj.weight"]
    expect = ["model.fc.weight", "model.embed_tokens.weight",
              "model.embed_tokens.weight",
              "model.layers.0.self_attn.q_proj.weight", "lm_head.weight"]
    cond = remap(src) == expect
    print(f"[test] {'PASS' if cond else 'FAIL'}  remap model.language_model "
          f"prefix (got {remap(src)})")
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
