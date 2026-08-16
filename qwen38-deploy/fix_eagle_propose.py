#!/usr/bin/env python
"""把 vllm-kunlun eagle.py 的 propose 签名对齐 vllm 0.15.1 的 10 参数调用。

vllm 0.15.1 的 gpu_model_runner.py 调用 propose 时传 10 个参数
（mm_embed_inputs + num_rejected_tokens_gpu + slot_mappings），而
vllm-kunlun 的 propose 是旧版 8 参数（mm_embeds），导致运行时
`propose() got an unexpected keyword argument 'mm_embed_inputs'`。

修复：签名把 mm_embeds 改成 mm_embed_inputs，并补齐 num_rejected_tokens_gpu
与 slot_mappings（默认 None），函数体内同步替换。

Usage: python fix_eagle_propose.py --apply <eagle.py>
       python fix_eagle_propose.py --test <site-packages-root>
"""
import sys

OLD_SIG = """    last_token_indices: Optional[torch.Tensor],
    common_attn_metadata: CommonAttentionMetadata,
    sampling_metadata: SamplingMetadata,
    mm_embeds: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:"""

NEW_SIG = """    last_token_indices: Optional[torch.Tensor],
    common_attn_metadata: CommonAttentionMetadata,
    sampling_metadata: SamplingMetadata,
    mm_embed_inputs: Optional[tuple[list[torch.Tensor], torch.Tensor]] = None,
    num_rejected_tokens_gpu: Optional[torch.Tensor] = None,
    slot_mappings: Optional[dict[str, torch.Tensor]] = None,
) -> torch.Tensor:"""

OLD_BODY = "            multimodal_embeddings=mm_embeds or None,"

NEW_BODY = "            multimodal_embeddings=mm_embed_inputs or None,"


def apply(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    n_sig = src.count(OLD_SIG)
    n_body = src.count(OLD_BODY)
    assert n_sig == 1, f"old sig occurrences = {n_sig} (expected 1)"
    assert n_body == 1, f"old body occurrences = {n_body} (expected 1)"
    src = src.replace(OLD_SIG, NEW_SIG).replace(OLD_BODY, NEW_BODY)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[apply] patched {path} OK")


def test(site_packages: str) -> None:
    p = f"{site_packages}/vllm_kunlun/v1/sample/spec_decode/eagle.py"
    src = open(p, encoding="utf-8").read()
    assert "mm_embed_inputs: Optional[tuple[list[torch.Tensor], torch.Tensor]]" in src, \
        "mm_embed_inputs signature not found"
    assert "num_rejected_tokens_gpu: Optional[torch.Tensor] = None," in src, \
        "num_rejected_tokens_gpu param not found"
    assert "multimodal_embeddings=mm_embed_inputs or None," in src, \
        "mm_embed_inputs body not found"
    print("[test] PASS  propose signature aligned to 10 args")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
