import time
import torch

def bench_torch(batch, iters=50):
    H, HV, K, V = 16, 48, 128, 128
    dev = "cuda"
    dtp = torch.float16
    q = torch.randn(batch, H, K, device=dev, dtype=dtp)
    k = torch.randn(batch, H, K, device=dev, dtype=dtp)
    v = torch.randn(batch, HV, V, device=dev, dtype=dtp)
    g = torch.randn(batch, HV, device=dev, dtype=dtp)
    beta = torch.randn(batch, HV, device=dev, dtype=dtp)
    S = torch.randn(batch, HV, K, V, device=dev, dtype=dtp)

    # expand k/q from H=16 to HV=48 (GVA: 3 v-heads per k-head)
    rep = HV // H
    k_e = k.repeat_interleave(rep, dim=1).contiguous()
    q_e = q.repeat_interleave(rep, dim=1).contiguous()
    g_exp = torch.exp(g.to(torch.float32)).to(dtp).unsqueeze(-1)
    beta_u = beta.unsqueeze(-1)

    def step():
        # S = S * exp(g)
        S2 = S * g_exp
        # kS = k @ S  -> [B,HV,V]
        kS = torch.einsum("bnk,bnkv->bnv", k_e, S2)
        # residual = v - kS
        res = v - kS
        # S += outer(k, res) * beta
        upd = torch.einsum("bnk,bnv->bnkv", k_e, res)
        S2 = S2 + upd * beta_u
        # o = q @ S
        o = torch.einsum("bnk,bnkv->bnv", q_e, S2)
        return o

    for _ in range(5):
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1000
    print(f"torch batch={batch:4d}: {dt:8.3f} ms, {batch/dt*1000:9.0f} tok/s", flush=True)

if __name__ == "__main__":
    for b in [1, 8, 32, 64, 128, 256]:
        bench_torch(b)
