import sys

p = sys.argv[1]
src = open(p).read()

# 1. imports: os + xspeedgate_ops + chunk lowrank
old = "import kunlun_ops\nimport torch\n"
assert old in src
src = src.replace(
    old,
    "import os\nimport xspeedgate_ops\nimport kunlun_ops\nimport torch\n",
    1,
)

old = "from vllm_kunlun.ops.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update\n"
assert old in src
src = src.replace(
    old,
    old + "from vllm_kunlun.ops.fla.chunk import chunk_gated_delta_rule_lowrank\n",
    1,
)

# 2. helper after logger
anchor = "logger = init_logger(__name__)"
assert anchor in src
src = src.replace(
    anchor,
    anchor + '''

_GDN_LOWRANK = os.getenv("LOWRANK_GDN") == "1"
_GDN_LOWRANK_R = 16
''',
    1,
)

# 3. slot_state
old = "        ssm_state = self_kv_cache[1]\n"
assert old in src
src = src.replace(
    old,
    old + "        slot_state = self_kv_cache[2] if _GDN_LOWRANK else None\n",
    1,
)

# 4. spec recurrent branch
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

# 5. prefill branch
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
                num_seqs = non_spec_query_start_loc.shape[0] - 1
                n_heads_v = value_non_spec.shape[-2]
                k_dim = key_non_spec.shape[-1]
                v_dim = value_non_spec.shape[-1]
                uv0 = torch.zeros(
                    num_seqs, n_heads_v, k_dim + v_dim, _GDN_LOWRANK_R,
                    device=value_non_spec.device, dtype=torch.float32,
                )
                slot0 = torch.zeros(
                    num_seqs, n_heads_v,
                    device=value_non_spec.device, dtype=torch.int32,
                )
                core_attn_out_non_spec, uv_final, slot_final = (
                    chunk_gated_delta_rule_lowrank(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        scale=None,
                        initial_uv=uv0,
                        initial_slot=slot0,
                        cu_seqlens=non_spec_query_start_loc,
                        cu_seqlens_cpu=non_spec_query_start_loc_cpu,
                        use_qk_l2norm_in_kernel=True,
                        rank=_GDN_LOWRANK_R,
                    )
                )
                uv_flat = uv_final.reshape(
                    uv_final.shape[0], -1, uv_final.shape[-1]
                ).to(ssm_state.dtype)
                cast_ssm_state = ssm_state.view(
                    ssm_state.shape[0], 1, -1, ssm_state.shape[-1]
                )
                kunlun_ops.reshape_and_cache_flash(
                    uv_flat,
                    uv_flat,
                    cast_ssm_state,
                    cast_ssm_state,
                    non_spec_state_indices_tensor,
                )
                slot_flat = slot_final[:, :1].contiguous()
                cast_slot = slot_state.view(
                    slot_state.shape[0], 1, -1
                )
                kunlun_ops.reshape_and_cache_flash(
                    slot_flat,
                    slot_flat,
                    cast_slot,
                    cast_slot,
                    non_spec_state_indices_tensor,
                )
                last_recurrent_state = None
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

# 6. decode non-spec branch
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
