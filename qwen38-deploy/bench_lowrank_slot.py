import torch
import kunlun_ops
import xspeedgate_ops

H, HV, K, V, r = 16, 48, 128, 128, 16
T, batch = 1, 1
dev, dtp = "cuda", torch.float16
scale = 1.0 / (K ** 0.5)
cu = torch.arange(batch + 1, device=dev, dtype=torch.int32)
idx = torch.arange(batch * T, device=dev, dtype=torch.int32)
torch.manual_seed(0)

# Build a full-rank state via 16 full-rank recurrent steps (simulating prefill)
h_full = torch.zeros(batch, HV, K, V, device=dev, dtype=dtp)
for t in range(16):
    q = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
    k = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
    v = torch.randn(batch, T, HV, V, device=dev, dtype=dtp)
    g = torch.randn(batch, T, HV, device=dev, dtype=dtp)
    b = torch.randn(batch, T, HV, device=dev, dtype=dtp)
    o, h_full = kunlun_ops.fused_recurrent_gated_delta_rule_fwdv2(
        q, k, v, g, b, scale, h_full,
        inplace_final_state=True, cu_seqlens=cu, h0_indices=idx,
        num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
        is_h0_transposed=True)

# Next token: full-rank reference
q_n = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
k_n = torch.randn(batch, T, H, K, device=dev, dtype=dtp)
v_n = torch.randn(batch, T, HV, V, device=dev, dtype=dtp)
g_n = torch.randn(batch, T, HV, device=dev, dtype=dtp)
b_n = torch.randn(batch, T, HV, device=dev, dtype=dtp)
o_ref, _ = kunlun_ops.fused_recurrent_gated_delta_rule_fwdv2(
    q_n, k_n, v_n, g_n, b_n, scale, h_full,
    inplace_final_state=True, cu_seqlens=cu, h0_indices=idx,
    num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
    is_h0_transposed=True)
torch.cuda.synchronize()

# Project full-rank state to UV (same random projection as the patch)
def project(S):
    Sf = S.float()
    omega = torch.randn(Sf.shape[-1], r, device=Sf.device, dtype=torch.float32)
    U = torch.matmul(Sf, omega)
    Vt = torch.matmul(U.transpose(-1, -2), Sf)
    UV = torch.cat([U, Vt.transpose(-1, -2)], dim=-2)
    return UV.to(S.dtype).contiguous()

UV = project(h_full)

for slot_init in [0, r - 1, r, r + 1]:
    UVc = UV.clone()
    slot = torch.full((batch,), slot_init, device=dev, dtype=torch.int32)
    o_lr = torch.zeros_like(v_n)
    try:
        torch.ops.xspeedgate_ops.fused_recurrent_gated_delta_rule_lowrank_fwd(
            q_n, k_n, v_n, g_n, b_n, o_lr, UVc, False, True, scale, cu, None,
            UVc, idx, slot)
        torch.cuda.synchronize()
        diff = (o_ref - o_lr).abs().max().item()
        print(f"slot_init={slot_init}: max|diff|={diff:.6f}")
    except Exception as e:
        print(f"slot_init={slot_init}: ERROR {e}")
