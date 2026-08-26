import torch
import xspeedgate_ops
from vllm_kunlun.ops.fla.index import prepare_chunk_indices, prepare_chunk_offsets

H, HV, K, V, r = 16, 48, 128, 128, 16
B, T = 1, 128  # two chunks
dev, dtp = "cuda", torch.float16
scale = K ** -0.5
torch.manual_seed(0)

q = torch.randn(B, T, H, K, device=dev, dtype=dtp)
k = torch.randn(B, T, H, K, device=dev, dtype=dtp)
k = k / k.norm(dim=-1, keepdim=True)
v = torch.randn(B, T, HV, V, device=dev, dtype=dtp)
g = -torch.rand(B, T, HV, device=dev, dtype=dtp) * 3.0  # log-space negative
beta = torch.rand(B, T, HV, device=dev, dtype=dtp).sigmoid()
cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
chunk_size = 64
ci = prepare_chunk_indices(cu, chunk_size)
co = prepare_chunk_offsets(cu, chunk_size)

def pipeline(k, v, beta, g, cu, ci, co):
    g2 = torch.ops.xspeedgate_ops.chunk_local_cumsum(
        g, chunk_size=chunk_size, reverse=False, cu_seqlens=cu, chunk_indices=ci, head_first=False)
    A = torch.ops.xspeedgate_ops.chunk_scaled_dot_kkt_fwd(
        k, beta, g2, cu, ci, chunk_size)
    torch.ops.xspeedgate_ops.solve_tril_ns(A, cu, ci, chunk_size)
    w, u = torch.ops.xspeedgate_ops.recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=g2, cu_seqlens=cu, chunk_indices=ci, chunk_size=chunk_size)
    return g2, u, w

g2, u, w = pipeline(k, v, beta, g, cu, ci, co)

# full-rank (zero initial state)
h0 = torch.zeros(1, HV, K, V, device=dev, dtype=dtp)
h, v_new, final_state = torch.ops.xspeedgate_ops.chunk_gated_delta_rule_fwd_h(
    k, u, w, g2, h0, cu, ci, co.to(torch.int32), chunk_size, True, True)
o_ref = torch.ops.xspeedgate_ops.chunk_fwd_o(
    q=q, k=k, v=v_new, h=h, g=g2, scale=scale, cu_seqlens=cu, chunk_indices=ci, chunk_size=chunk_size)
torch.cuda.synchronize()
print("full-rank:", o_ref.shape, "final", final_state.shape)

# low-rank (zero initial uv + slot)
uv0 = torch.zeros(1, HV, K + V, r, device=dev, dtype=torch.float32)
slot0 = torch.zeros(1, HV, device=dev, dtype=torch.int32)
UV_chunks, UV_final, slot_final, v_new_lr = torch.ops.xspeedgate_ops.chunk_gated_delta_rule_lowrank_fwd_h(
    k, u, w, g2, uv0, slot0, cu, ci, co.to(torch.int32), chunk_size, r)
torch.cuda.synchronize()
print("lowrank UV_chunks", UV_chunks.shape, "UV_final", UV_final.shape, "slot", slot_final, "v_new", v_new_lr.shape)
diff_v = (v_new - v_new_lr).abs().max().item()
print(f"v_new diff: {diff_v:.6f}")

# reconstruct h from UV_chunks and feed chunk_fwd_o
Uc = UV_chunks[..., :K, :]   # [B, NT, HV, K, r]
Vc = UV_chunks[..., K:, :]   # [B, NT, HV, V, r]
h_lr = torch.matmul(Uc, Vc.transpose(-1, -2))  # [B, NT, HV, K, V]
h_lr = h_lr.to(dtp)
o_lr = torch.ops.xspeedgate_ops.chunk_fwd_o(
    q=q, k=k, v=v_new_lr, h=h_lr, g=g2, scale=scale, cu_seqlens=cu, chunk_indices=ci, chunk_size=chunk_size)
torch.cuda.synchronize()
diff_o = (o_ref - o_lr).abs()
print(f"o diff: max={diff_o.max().item():.6f} mean={diff_o.nanmean().item():.6f}")
hdiff = (h - h_lr).abs()
print(f"h diff: max={hdiff.max().item():.6f}")

# also compare final state: full-rank final vs UV_final reconstruction
Uf = UV_final[..., :K, :]
Vf = UV_final[..., K:, :]
final_lr = torch.matmul(Uf, Vf.transpose(-1, -2))
print("full final:", final_state.flatten()[:4].tolist(), "absmax", final_state.abs().max().item())
print("lr final:", final_lr.flatten()[:4].tolist(), "absmax", final_lr.abs().max().item())
diff_f = (final_state - final_lr.to(final_state.dtype)).abs().max().item()
print(f"final_state diff: {diff_f:.6f}")
