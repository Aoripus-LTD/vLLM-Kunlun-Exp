import os
import time
import torch
import kunlun_ops

def bench(batch, iters=200, fast_opt=None):
    H, HV, K, V = 16, 48, 128, 128
    T = 1
    dev = "cuda"
    dtp = torch.float16
    q = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
    k = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
    v = torch.randn(batch, T, HV, V, device=dev, dtype=dtp)
    g = torch.randn(batch, T, HV, device=dev, dtype=dtp)
    beta = torch.randn(batch, T, HV, device=dev, dtype=dtp)
    scale = 1.0 / (K ** 0.5)
    h0 = torch.randn(batch, HV, K, V, device=dev, dtype=dtp)
    cu = torch.arange(batch + 1, device=dev, dtype=torch.int32)
    idx = torch.arange(batch * T, device=dev, dtype=torch.int32)

    if fast_opt is not None:
        os.environ["XPU_SET_RECURRENT_GATED_DELTA_RULE_FWDV2_FP16_FAST_OPT"] = str(fast_opt)
    else:
        os.environ.pop("XPU_SET_RECURRENT_GATED_DELTA_RULE_FWDV2_FP16_FAST_OPT", None)

    try:
        for _ in range(10):
            o, st = kunlun_ops.fused_recurrent_gated_delta_rule_fwdv2(
                q, k, v, g, beta, scale, h0,
                inplace_final_state=True, cu_seqlens=cu, h0_indices=idx,
                num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
                is_h0_transposed=True,
            )
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            o, st = kunlun_ops.fused_recurrent_gated_delta_rule_fwdv2(
                q, k, v, g, beta, scale, h0,
                inplace_final_state=True, cu_seqlens=cu, h0_indices=idx,
                num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
                is_h0_transposed=True,
            )
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1000
        print(f"batch={batch} fast_opt={fast_opt}: {dt:.3f} ms/kernel, {batch/dt*1000:.0f} tok/s", flush=True)
        return o
    except Exception as e:
        print(f"batch={batch} fast_opt={fast_opt}: ERROR {e}", flush=True)
        return None

if __name__ == "__main__":
    outs = {}
    for fo in [None, 0, 1, 2, 3]:
        outs[str(fo)] = bench(1, fast_opt=fo)
    # compare outputs
    ref = outs.get("None")
    if ref is not None:
        for k, v in outs.items():
            if v is not None and k != "None":
                diff = (ref - v).abs().max().item()
                print(f"output diff vs opt=None for fast_opt={k}: {diff:.6f}", flush=True)
