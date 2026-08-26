import time
import torch
import xspeedgate_ops

# Step 1: SVD on XPU
print("=== SVD on XPU ===")
for n in [1, 4]:
    S = torch.randn(n, 48, 128, 128, device="cuda", dtype=torch.float16)
    Sf = S.float()
    t0 = time.perf_counter()
    try:
        U, Sh, Vh = torch.linalg.svd(Sf)
        ok = True
    except Exception as e:
        ok = False
        print(f"torch.linalg.svd n={n}: FAIL {e}")
    torch.cuda.synchronize()
    if ok:
        dt = time.perf_counter() - t0
        print(f"torch.linalg.svd n={n}: {dt*1000:.1f} ms")
    # try svd_lowrank
    try:
        t0 = time.perf_counter()
        U, Sh, Vh = torch.svd_lowrank(Sf, q=16, niter=2)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"torch.svd_lowrank n={n} q=16: {dt*1000:.1f} ms")
    except Exception as e:
        print(f"torch.svd_lowrank n={n}: FAIL {e}")

# Step 2: lowrank in-place semantics (UV0_source + slot_state), 2 steps
print("=== lowrank in-place 2-step ===")
H, HV, K, V, r = 16, 48, 128, 128, 16
batch, T = 1, 1
dev, dtp = "cuda", torch.float16
torch.manual_seed(0)
scale = 1.0 / (K ** 0.5)
cu = torch.arange(batch + 1, device=dev, dtype=torch.int32)
idx = torch.arange(batch * T, device=dev, dtype=torch.int32)

# init from zero, then step1 (write into UV0), step2 (continue with slot)
UV0 = torch.zeros(batch, HV, K + V, r, device=dev, dtype=dtp)
slot = torch.zeros(batch, device=dev, dtype=torch.int32)

o = torch.zeros(batch, T, HV, V, device=dev, dtype=dtp)
for step in range(3):
    q = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
    k = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
    v = torch.randn(batch, T, HV, V, device=dev, dtype=dtp)
    g = torch.randn(batch, T, HV, device=dev, dtype=dtp)
    beta = torch.randn(batch, T, HV, device=dev, dtype=dtp)
    torch.ops.xspeedgate_ops.fused_recurrent_gated_delta_rule_lowrank_fwd(
        q, k, v, g, beta, o, UV0, False, True, scale, cu, None,
        UV0, idx, slot)
    torch.cuda.synchronize()
    print(f"step={step}: o[0,0,0,:4]={o[0,0,0,:4].tolist()} slot={slot.tolist()} UV0[0,0,0,0]={UV0[0,0,0,0].item():.4f}")
