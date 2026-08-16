#!/bin/bash
# verify_ptrview.sh — 验证 ptr-view + copy_ 方案（torch._C._construct_storage_from_data_pointer）
/opt/vllm_kunlun/bin/python - <<'PYEOF'
import torch

print("hasattr _construct_storage_from_data_pointer:",
      hasattr(torch._C, "_construct_storage_from_data_pointer"))

def make_view(ptr, size, device):
    storage = torch._C._construct_storage_from_data_pointer(ptr, device, size)
    t = torch.empty(0, dtype=torch.uint8, device=device)
    return t.set_(storage, 0, (size,), (1,))

src = torch.randn(64, device="cuda", dtype=torch.float32)
dst = torch.zeros(64, device="cuda", dtype=torch.float32)
n = 4
src_ptrs = torch.tensor(
    [src.data_ptr() + i * 16 * 4 for i in range(n)], device="cuda", dtype=torch.int64)
dst_ptrs = torch.tensor(
    [dst.data_ptr() + i * 16 * 4 for i in range(n)], device="cuda", dtype=torch.int64)
sizes = torch.tensor([16 * 4] * n, device="cuda", dtype=torch.int32)

s_cpu = src_ptrs.detach().cpu().tolist()
d_cpu = dst_ptrs.detach().cpu().tolist()
z_cpu = sizes.detach().cpu().tolist()
for sptr, dptr, size in zip(s_cpu, d_cpu, z_cpu):
    sv = make_view(sptr, size, src.device)
    dv = make_view(dptr, size, src.device)
    dv.copy_(sv, non_blocking=True)
torch.cuda.synchronize()
print("copy 结果正确:", torch.equal(src, dst))
print("src[:8] =", src[:8].tolist())
print("dst[:8] =", dst[:8].tolist())
PYEOF
