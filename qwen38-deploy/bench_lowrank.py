import time
import torch
import xspeedgate_ops

def bench(batch, rank, iters=100):
    H, HV, K, V = 16, 48, 128, 128
    T = 1
    dev = "cuda"
    dtp = torch.float16
    q = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
    k = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
    v = torch.randn(batch, T, HV, V, device=dev, dtype=dtp)
    g = torch.randn(batch, T, HV, device=dev, dtype=dtp)
    beta = torch.randn(batch, T, HV, device=dev, dtype=dtp)
    o = torch.zeros(batch, T, HV, V, device=dev, dtype=dtp)
    UVt = torch.zeros(batch, HV, K + V, rank, device=dev, dtype=dtp)
    cu = torch.arange(batch + 1, device=dev, dtype=torch.int32)
    scale = 1.0 / (K ** 0.5)

    try:
        for _ in range(5):
            torch.ops.xspeedgate_ops.fused_recurrent_gated_delta_rule_lowrank_fwd(
                q, k, v, g, beta, o, UVt, False, True, scale, cu, None, None, None, None)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.ops.xspeedgate_ops.fused_recurrent_gated_delta_rule_lowrank_fwd(
                q, k, v, g, beta, o, UVt, False, True, scale, cu, None, None, None, None)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1000
        print(f"batch={batch:4d} rank={rank:3d}: {dt:8.3f} ms, {batch/dt*1000:9.0f} tok/s", flush=True)
    except Exception as e:
        print(f"batch={batch:4d} rank={rank:3d}: ERROR {e}", flush=True)

if __name__ == "__main__":
    for r in [16, 32, 64]:
        for b in [1, 8, 64, 256]:
            bench(b, r)
