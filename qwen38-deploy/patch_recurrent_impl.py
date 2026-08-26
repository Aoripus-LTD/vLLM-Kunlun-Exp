import sys

p = sys.argv[1]
src = open(p).read()

old = """        o, final_state = kunlun_ops.fused_recurrent_gated_delta_rule_fwdv2(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            scale,
            initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            h0_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            is_h0_transposed=True,
        )"""

new = """        B, T, H, K = k.shape
        HV = v.shape[2]
        o = torch.zeros_like(v)
        if inplace_final_state:
            ht_output = initial_state
        else:
            ht_output = q.new_empty(B, HV, K, v.shape[-1], dtype=torch.float32)
        is_beta_headwise = beta.ndim == v.ndim
        torch.ops.xspeedgate_ops.fused_recurrent_gated_delta_rule_fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous().to(q.dtype),
            beta.contiguous().to(q.dtype),
            o,
            ht_output,
            is_beta_headwise,
            use_qk_l2norm_in_kernel,
            inplace_final_state,
            True,
            scale,
            cu_seqlens,
            num_accepted_tokens,
            initial_state,
            ssm_state_indices,
        )
        final_state = ht_output"""

assert old in src, "kunlun call not found"
src = src.replace(old, new, 1)
open(p + ".sw", "w").write(src)
print("written", p + ".sw")
