#!/bin/bash
# verify_xspeed_batchmemcpy.sh — 验证 xspeedgate_ops.batch_memcpy 存在且可用
/opt/vllm_kunlun/bin/python - <<'PYEOF'
import torch

print("hasattr batch_memcpy:", hasattr(torch.ops.xspeedgate_ops, "batch_memcpy"))
if not hasattr(torch.ops.xspeedgate_ops, "batch_memcpy"):
    raise SystemExit(1)

# 小规模功能测试：src 拷入 dst（与 mamba state copy 同构）
src = torch.randn(64, device="cuda", dtype=torch.float32)
dst = torch.zeros(64, device="cuda", dtype=torch.float32)
n = 4  # 分 4 块拷贝，每块 16 元素
src_ptrs = torch.tensor(
    [src.data_ptr() + i * 16 * 4 for i in range(n)], device="cuda", dtype=torch.int64)
dst_ptrs = torch.tensor(
    [dst.data_ptr() + i * 16 * 4 for i in range(n)], device="cuda", dtype=torch.int64)
sizes = torch.tensor([16 * 4] * n, device="cuda", dtype=torch.int32)
torch.ops.xspeedgate_ops.batch_memcpy(src_ptrs, dst_ptrs, sizes)
torch.cuda.synchronize()
ok = torch.equal(src, dst)
print("copy 结果正确:", ok)
print("src[:8] =", src[:8].tolist())
print("dst[:8] =", dst[:8].tolist())
PYEOF
