#!/bin/bash
# fix_mamba_copy.sh — 替换 vllm 原生 mamba_utils.py 的 batch_memcpy（triton → ptr-view copy_）
# 背景: prefix-caching 开启时 preprocess_mamba 触发 batch_memcpy_kernel（triton），
#       昆仑芯 triton 3.0.0 load_binary 报 CUDA_ERROR_NOT_SUPPORTED。
#       方案 = vllm_kunlun 注释保留的备胎: torch._C._construct_storage_from_data_pointer + copy_
P=/opt/vllm_kunlun/lib/python3.10/site-packages/vllm/v1/worker/mamba_utils.py
cp "$P" "$P.bak_mamba"
/opt/vllm_kunlun/bin/python - <<'PYEOF'
p = "/opt/vllm_kunlun/lib/python3.10/site-packages/vllm/v1/worker/mamba_utils.py"
s = open(p).read()
old = """def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    BLOCK_SIZE = 1024
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)
"""
new = """def _make_uint8_view_from_ptr(ptr, size, device):
    storage = torch._C._construct_storage_from_data_pointer(ptr, device, size)
    tensor = torch.empty(0, dtype=torch.uint8, device=device)
    return tensor.set_(storage, 0, (size,), (1,))


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch
    if batch == 0:
        return
    device = src_ptrs.device
    src_ptrs_cpu = src_ptrs.detach().cpu().tolist()
    dst_ptrs_cpu = dst_ptrs.detach().cpu().tolist()
    sizes_cpu = sizes.detach().cpu().tolist()
    for src_ptr, dst_ptr, size in zip(src_ptrs_cpu, dst_ptrs_cpu, sizes_cpu):
        if size <= 0 or src_ptr == dst_ptr:
            continue
        src = _make_uint8_view_from_ptr(src_ptr, size, device)
        dst = _make_uint8_view_from_ptr(dst_ptr, size, device)
        dst.copy_(src, non_blocking=True)
"""
assert old in s, "batch_memcpy 原文未匹配，中止（防止重复 patch 或版本不符）"
s = s.replace(old, new)
open(p, "w").write(s)
print("PATCHED")
PYEOF
echo "== 验证: py_compile + batch_memcpy_kernel 引用 =="
/opt/vllm_kunlun/bin/python -m py_compile "$P" && echo "py_compile OK"
grep -n "batch_memcpy_kernel\[grid\]" "$P" || echo "triton 调用已移除"
grep -n "def batch_memcpy" "$P"
