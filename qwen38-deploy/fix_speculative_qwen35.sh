#!/bin/bash
# fix_speculative_qwen35.sh — speculative.py qwen3_5 MTP 分支重建 Qwen3NextConfig
# 第 17 雷：Qwen3_5Config（多模态 wrapper）无 vocab_size/hidden_size，
# Qwen3NextMultiTokenPredictor.__init__ 直接 AttributeError。
# 修复：qwen3_5 分支用 text 子配置重建真正的 Qwen3NextConfig 实例。
set -e
T=/opt/vllm_kunlun/lib/python3.10/site-packages/vllm/config/speculative.py
PY=/opt/vllm_kunlun/bin/python

cp "$T" "$T.bak_rs"
echo "备份: $T.bak_rs"

"$PY" - "$T" <<'EOF'
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()

old = '''        if hf_config.model_type in ("qwen3_5", "qwen3_5_text"):
            # Kunlun port: Qwen3.8 multimodal model reuses the Qwen3Next MTP head
            hf_config.model_type = "qwen3_next"'''

new = '''        if hf_config.model_type in ("qwen3_5", "qwen3_5_text"):
            # Kunlun port: Qwen3.8 multimodal model reuses the Qwen3Next MTP
            # head. The multimodal Qwen3_5Config wraps text/vision sub-configs
            # and has no vocab_size/hidden_size of its own; rebuild a real
            # Qwen3NextConfig from the text sub-config so the MTP head gets
            # the fields it needs.
            from vllm.transformers_utils.configs import Qwen3NextConfig
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
            hf_config.model_type = "qwen3_next"'''

assert src.count(old) == 1, f"pattern not found or not unique (count={src.count(old)})"
src = src.replace(old, new)
open(path, "w", encoding="utf-8").write(src)
print("patched:", path)
EOF

echo "== 验证 1: py_compile =="
"$PY" -m py_compile "$T" && echo "语法 OK"
echo "== 验证 2: patch 存在 =="
grep -n "Qwen3NextConfig" "$T" | head -5
echo "== 验证 3: 重建逻辑实测（加载模型 config.json 模拟） =="
"$PY" - <<'EOF'
from transformers import AutoConfig
from vllm.transformers_utils.configs import Qwen3NextConfig

hf_config = AutoConfig.from_pretrained(
    "/home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic", trust_remote_code=True
)
print("原始 model_type:", hf_config.model_type, "| 类型:", type(hf_config).__name__)

# 复制 speculative.py patch 的修复逻辑
if hf_config.model_type in ("qwen3_5", "qwen3_5_text"):
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
    hf_config.model_type = "qwen3_next"

print("重建后 model_type:", hf_config.model_type, "| 类型:", type(hf_config).__name__)
print("vocab_size:", hf_config.vocab_size)
print("hidden_size:", hf_config.hidden_size)
print("num_hidden_layers:", hf_config.num_hidden_layers)
print("num_attention_heads:", hf_config.num_attention_heads)
print("num_key_value_heads:", hf_config.num_key_value_heads)
print("intermediate_size:", hf_config.intermediate_size)
print("max_position_embeddings:", hf_config.max_position_embeddings)
print("rms_norm_eps:", hf_config.rms_norm_eps)
print("layer_types:", getattr(hf_config, "layer_types", None))
print("linear_conv_kernel_dim:", getattr(hf_config, "linear_conv_kernel_dim", None))
assert hf_config.vocab_size == 248320, "vocab_size 不符"
assert hf_config.hidden_size == 4096, "hidden_size 不符"
print("单测通过")
EOF
echo "全部通过"
