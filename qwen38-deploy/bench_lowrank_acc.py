import torch
import kunlun_ops
import xspeedgate_ops

H, HV, K, V = 16, 48, 128, 128
T, batch = 1, 1
dev, dtp = "cuda", torch.float16

torch.manual_seed(0)
q = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
k = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
v = torch.randn(batch, T, HV, V, device=dev, dtype=dtp)
g = torch.randn(batch, T, HV, device=dev, dtype=dtp)
beta = torch.randn(batch, T, HV, device=dev, dtype=dtp)
scale = 1.0 / (K ** 0.5)
h0 = torch.zeros(batch, HV, K, V, device=dev, dtype=dtp)
cu = torch.arange(batch + 1, device=dev, dtype=torch.int32)
idx = torch.arange(batch * T, device=dev, dtype=torch.int32)

# full-rank reference (zero init)
o_ref, st_ref = kunlun_ops.fused_recurrent_gated_delta_rule_fwdv2(
    q, k, v, g, beta, scale, h0,
    inplace_final_state=True, cu_seqlens=cu, h0_indices=idx,
    num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
    is_h0_transposed=True,
)
torch.cuda.synchronize()

for r in [16, 32]:
    UVt = torch.zeros(batch, HV, K + V, r, device=dev, dtype=dtp)
    o_lr = torch.zeros_like(v)
    torch.ops.xspeedgate_ops.fused_recurrent_gated_delta_rule_lowrank_fwd(
        q, k, v, g, beta, o_lr, UVt, False, True, scale, cu, None, None, None, None)
    torch.cuda.synchronize()
    diff = (o_ref - o_lr).abs()
    rel = diff.mean().item() / (o_ref.abs().mean().item() + 1e-6)
    print(f"rank={r} zero-init: max|diff|={diff.max().item():.6f} mean|diff|={diff.mean().item():.6f} rel={rel:.6f}")

# multi-step accumulation comparison (state divergence over 8 steps)
for r in [16, 32]:
    h_full = torch.zeros(batch, HV, K, V, device=dev, dtype=dtp)
    UVt = torch.zeros(batch, HV, K + V, r, device=dev, dtype=dtp)
    slot = torch.zeros(batch, device=dev, dtype=torch.int32)
    maxdiff = 0.0
    for t in range(8):
        q_t = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
        k_t = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
        v_t = torch.randn(batch, T, HV, V, device=dev, dtype=dtp)
        g_t = torch.randn(batch, T, HV, device=dev, dtype=dtp)
        b_t = torch.randn(batch, T, HV, device=dev, dtype=dtp)
        o_ref, st = kunlun_ops.fused_recurrent_gated_delta_rule_fwdv2(
            q_t, k_t, v_t, g_t, b_t, scale, h_full,
            inplace_final_state=True, cu_seqlens=cu, h0_indices=idx,
            num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
            is_h0_transposed=True)
        o_lr = torch.zeros_like(v_t)
        torch.ops.xspeedgate_ops.fused_recurrent_gated_delta_rule_lowrank_fwd(
            q_t, k_t, v_t, g_t, b_t, o_lr, UVt, False, True, scale, cu, None,
            UVt, idx, slot)
        torch.cuda.synchronize()
        maxdiff = max(maxdiff, (o_ref - o_lr).abs().max().item())
    print(f"rank={r} 8-step stateful: max|diff|={maxdiff:.6f}")
