import sys

p = sys.argv[1]
src = open(p).read()

anchor = "def chunk_gated_delta_rule("
assert anchor in src
func = '''def chunk_gated_delta_rule_lowrank(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_uv: torch.Tensor = None,
    initial_slot: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    rank: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Low-rank chunk Gated Delta Rule prefill (xspeedgate_ops).

    State is kept as UV [N, HV, K+V, rank] with a per-head slot counter.
    Returns (o, UV_final, slot_final). Caller writes UV_final into the
    paged ssm_state cache and slot_final[:, 0] into the slot buffer.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    chunk_size = 64
    cu_cpu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()
    ci = prepare_chunk_indices(cu_cpu, chunk_size).to(q.device)
    co = prepare_chunk_offsets(cu_cpu, chunk_size).to(q.device)

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    g_c = torch.ops.xspeedgate_ops.chunk_local_cumsum(
        g, chunk_size=chunk_size, reverse=False,
        cu_seqlens=cu_seqlens, chunk_indices=ci, head_first=False,
    )
    A = torch.ops.xspeedgate_ops.chunk_scaled_dot_kkt_fwd(
        k, beta, g_c, cu_seqlens, ci, chunk_size,
    )
    torch.ops.xspeedgate_ops.solve_tril_ns(A, cu_seqlens, ci, chunk_size)
    w, u = torch.ops.xspeedgate_ops.recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=g_c,
        cu_seqlens=cu_seqlens, chunk_indices=ci, chunk_size=chunk_size,
    )
    UV_chunks, UV_final, slot_final, v_new = (
        torch.ops.xspeedgate_ops.chunk_gated_delta_rule_lowrank_fwd_h(
            k, u, w, g_c, initial_uv, initial_slot,
            cu_seqlens, ci, co.to(torch.int32), chunk_size, rank,
        )
    )
    K = q.shape[-1]
    Uc = UV_chunks[..., :K, :]
    Vc = UV_chunks[..., K:, :]
    h_lr = torch.matmul(Uc, Vc.transpose(-1, -2))
    o = torch.ops.xspeedgate_ops.chunk_fwd_o(
        q=q, k=k, v=v_new, h=h_lr, g=g_c, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=ci, chunk_size=chunk_size,
    )
    return o, UV_final, slot_final


'''
src = src.replace(anchor, func + anchor, 1)
open(p + ".lr", "w").write(src)
print("patched", p)
