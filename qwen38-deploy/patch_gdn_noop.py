import re
import sys

# usage: patch_gdn_noop.py <qwen3_next.py> <noop_conv> <noop_rec>
# noop_conv / noop_rec: "1" to skip the kernel, "0" to keep
p = sys.argv[1]
noop_conv = sys.argv[2] == "1"
noop_rec = sys.argv[3] == "1"
src = open(p).read()

if noop_conv:
    old = """        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_decodes
                ],
                conv_state_indices_cpu=non_spec_state_indices_tensor_cpu[
                    : attn_metadata.num_decodes
                ],
                validate_data=True,
            )"""
    new = """        elif attn_metadata.num_decodes > 0:
            # NOOP-CONV for profiling: skip causal_conv1d_update
            mixed_qkv_non_spec = mixed_qkv_non_spec"""
    assert old in src, "conv decode branch not found"
    src = src.replace(old, new, 1)

if noop_rec:
    old = """            core_attn_out_non_spec, last_recurrent_state = (
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
    new = """            # NOOP-REC for profiling: skip fused_recurrent_gated_delta_rule
            core_attn_out_non_spec = value_non_spec
            last_recurrent_state = None"""
    assert old in src, "rec decode branch not found"
    src = src.replace(old, new, 1)

open(p + ".noop", "w").write(src)
print(f"written {p}.noop (conv={noop_conv}, rec={noop_rec})")
