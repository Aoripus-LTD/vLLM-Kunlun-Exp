#!/usr/bin/env python
"""把 vllm-kunlun eagle.py 的 prepare_next_token_ids_padded 改回 5 参数 mask。

vllm-kunlun 从新版本移植的 eagle.py 把 prepare_next_token_ids_padded 改成
6 参数（discard_request_indices + num_discarded_requests），但 vllm 0.15.1
的 gpu_model_runner.py 仍按 5 参数调用（discard_request_mask bool mask），
导致运行时 `missing 1 required positional argument: 'num_discarded_requests'`。

修复：签名改回 5 参数 discard_request_mask，函数内用 nonzero() 从 mask
推导 indices 与 count，保持原有 index_fill_ 逻辑不变。

Usage: python fix_eagle_padded.py --apply <eagle.py>
       python fix_eagle_padded.py --test <site-packages-root>
"""
import sys

OLD_SIG = """def prepare_next_token_ids_padded(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    sampled_token_ids: torch.Tensor,
    requests: dict[str, CachedRequestState],
    gpu_input_batch: InputBatch,
    discard_request_indices: torch.Tensor,
    num_discarded_requests: int,
) -> tuple[torch.Tensor, torch.Tensor]:"""

NEW_SIG = """def prepare_next_token_ids_padded(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    sampled_token_ids: torch.Tensor,
    requests: dict[str, CachedRequestState],
    gpu_input_batch: InputBatch,
    discard_request_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:"""

OLD_BODY = """    # Mask out the sampled tokens indices that should not be sampled.
    discard_sampled_tokens_req_indices = discard_request_indices[
        :num_discarded_requests
    ]"""

NEW_BODY = """    # Mask out the sampled tokens indices that should not be sampled.
    discard_request_indices = discard_request_mask.nonzero().flatten()
    num_discarded_requests = discard_request_indices.numel()
    discard_sampled_tokens_req_indices = discard_request_indices[
        :num_discarded_requests
    ]"""


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
    assert "discard_request_mask: torch.Tensor," in src, \
        "discard_request_mask signature not found"
    assert "discard_request_mask.nonzero().flatten()" in src, \
        "nonzero derivation not found"
    assert "num_discarded_requests: int," not in src, \
        "num_discarded_requests param still present"
    print("[test] PASS  prepare_next_token_ids_padded restored to 5-arg mask")
    print("[test] ALL PASS")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--apply":
        apply(sys.argv[2])
    elif mode == "--test":
        test(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
