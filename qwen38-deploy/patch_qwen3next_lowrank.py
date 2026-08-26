import sys

# patch_qwen3next_lowrank.py <qwen3_next.py>
# Env-gated LOWRANK_GDN=1: use low-rank UV state + slot buffer for GDN recurrent.

p = sys.argv[1]
src = open(p).read()

# 1. helper function after imports (anchor on "logger = init_logger")
anchor = "logger = init_logger(__name__)"
assert anchor in src
helper = anchor + '''

_GDN_LOWRANK = os.getenv("LOWRANK_GDN") == "1"
_GDN_LOWRANK_R = 16


def _fullrank_to_uv(S: torch.Tensor, r: int = _GDN_LOWRANK_R) -> torch.Tensor:
    """Project a full-rank GDN state [N, HV, K, V] to UV layout [N, HV, K+V, r].

    Random projection (two GEMMs) is used instead of SVD: torch.linalg.svd is
    impractically slow on XPU (~5s for one 128x128 stack). The projection is
    only applied once per sequence at the prefill->decode boundary.
    """
    Sf = S.float()
    omega = torch.randn(Sf.shape[-1], r, device=Sf.device, dtype=torch.float32)
    U = torch.matmul(Sf, omega)                      # [N, HV, K, r]
    Vt = torch.matmul(U.transpose(-1, -2), Sf)       # [N, HV, r, V]
    UV = torch.cat([U, Vt.transpose(-1, -2)], dim=-2)  # [N, HV, K+V, r]
    return UV.to(S.dtype).contiguous()
'''
src = src.replace(anchor, helper, 1)

# 2. slot_state after ssm_state
old = "        ssm_state = self_kv_cache[1]\n"
assert old in src
src = src.replace(
    old,
    old + "        slot_state = self_kv_cache[2] if _GDN_LOWRANK else None\n",
    1,
)

# 3. spec recurrent branch -> lowrank
old = """            tmp_core_attn_out_spec, last_recurrent_state = (
                fused_recurrent_gated_delta_rule(
                    q=query_spec[:, :actual_num],
                    k=key_spec[:, :actual_num],
                    v=value_spec[:, :actual_num],
                    g=g_spec[:, :actual_num],
                    beta=beta_spec[:, :actual_num],
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=tmp_cu_seqlens,
                    ssm_state_indices=spec_state_indices_tensor[
                        : attn_metadata.num_spec_decodes
                    ],
                    num_accepted_tokens=recurrent_num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
            core_attn_out_spec[:, :actual_num] = tmp_core_attn_out_spec"""
new = """            if _GDN_LOWRANK:
                torch.ops.xspeedgate_ops.fused_recurrent_gated_delta_rule_lowrank_fwd(
                    query_spec[:, :actual_num].contiguous(),
                    key_spec[:, :actual_num].contiguous(),
                    value_spec[:, :actual_num].contiguous(),
                    g_spec[:, :actual_num].contiguous(),
                    beta_spec[:, :actual_num].contiguous(),
                    core_attn_out_spec,
                    ssm_state,
                    False,
                    True,
                    None,
                    tmp_cu_seqlens,
                    recurrent_num_accepted_tokens,
                    ssm_state,
                    spec_state_indices_tensor[: attn_metadata.num_spec_decodes],
                    slot_state,
                )
            else:
                tmp_core_attn_out_spec, last_recurrent_state = (
                    fused_recurrent_gated_delta_rule(
                        q=query_spec[:, :actual_num],
                        k=key_spec[:, :actual_num],
                        v=value_spec[:, :actual_num],
                        g=g_spec[:, :actual_num],
                        beta=beta_spec[:, :actual_num],
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=tmp_cu_seqlens,
                        ssm_state_indices=spec_state_indices_tensor[
                            : attn_metadata.num_spec_decodes
                        ],
                        num_accepted_tokens=recurrent_num_accepted_tokens,
                        use_qk_l2norm_in_kernel=True,
                    )
                )
                core_attn_out_spec[:, :actual_num] = tmp_core_attn_out_spec"""
assert old in src, "spec branch missing"
src = src.replace(old, new, 1)

# 4. prefill branch -> project full-rank state to UV
old = """            # Init cache
            last_recurrent_state = (
                last_recurrent_state.transpose(-1, -2)
                .contiguous()
                .to(ssm_state.dtype)
                .view(last_recurrent_state.shape[0], -1, last_recurrent_state.shape[-1])
            )
            cast_ssm_state = ssm_state.view(
                ssm_state.shape[0], 1, -1, ssm_state.shape[-1]
            )

            kunlun_ops.reshape_and_cache_flash(
                last_recurrent_state,
                last_recurrent_state,
                cast_ssm_state,
                cast_ssm_state,
                non_spec_state_indices_tensor,
            )"""
new = """            # Init cache
            if _GDN_LOWRANK:
                last_recurrent_state = (
                    last_recurrent_state.transpose(-1, -2).contiguous()
                )
                uv_state = _fullrank_to_uv(last_recurrent_state)
                uv_state = uv_state.view(
                    uv_state.shape[0], -1, uv_state.shape[-1]
                ).to(ssm_state.dtype)
                cast_ssm_state = ssm_state.view(
                    ssm_state.shape[0], 1, -1, ssm_state.shape[-1]
                )
                kunlun_ops.reshape_and_cache_flash(
                    uv_state,
                    uv_state,
                    cast_ssm_state,
                    cast_ssm_state,
                    non_spec_state_indices_tensor,
                )
                # slot_state stays 0: a freshly projected UV state fills all r
                # slots, and the lowrank kernel wraps around from 0.
            else:
                last_recurrent_state = (
                    last_recurrent_state.transpose(-1, -2)
                    .contiguous()
                    .to(ssm_state.dtype)
                    .view(last_recurrent_state.shape[0], -1, last_recurrent_state.shape[-1])
                )
                cast_ssm_state = ssm_state.view(
                    ssm_state.shape[0], 1, -1, ssm_state.shape[-1]
                )

                kunlun_ops.reshape_and_cache_flash(
                    last_recurrent_state,
                    last_recurrent_state,
                    cast_ssm_state,
                    cast_ssm_state,
                    non_spec_state_indices_tensor,
                )"""
assert old in src, "prefill branch missing"
src = src.replace(old, new, 1)

# 5. decode non-spec branch -> lowrank
old = """        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_recurrent_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )"""
new = """        elif attn_metadata.num_decodes > 0:
            if _GDN_LOWRANK:
                core_attn_out_non_spec = torch.zeros_like(value_non_spec)
                torch.ops.xspeedgate_ops.fused_recurrent_gated_delta_rule_lowrank_fwd(
                    query_non_spec.contiguous(),
                    key_non_spec.contiguous(),
                    value_non_spec.contiguous(),
                    g_non_spec.contiguous(),
                    beta_non_spec.contiguous(),
                    core_attn_out_non_spec,
                    ssm_state,
                    False,
                    True,
                    None,
                    non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    None,
                    ssm_state,
                    non_spec_state_indices_tensor,
                    slot_state,
                )
                last_recurrent_state = None
            else:
                core_attn_out_non_spec, last_recurrent_state = (
                    fused_recurrent_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=non_spec_query_start_loc[
                            : attn_metadata.num_decodes + 1
                        ],
                        ssm_state_indices=non_spec_state_indices_tensor,
                        use_qk_l2norm_in_kernel=True,
                    )
                )"""
assert old in src, "decode branch missing"
src = src.replace(old, new, 1)

open(p + ".lr", "w").write(src)
print("patched", p)
