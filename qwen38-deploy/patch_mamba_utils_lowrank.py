import sys

# patch_mamba_utils_lowrank.py <mamba_utils.py>
# Adds env-gated low-rank GDN state layout:
#   LOWRANK_GDN=1 -> temporal state becomes [HV, K+V, r] + slot [1] (3 buffers)

p = sys.argv[1]
src = open(p).read()

# 1. dtype: add int32 slot buffer
old = '''    @classmethod
    def gated_delta_net_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, torch.dtype]:
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (state_dtype, state_dtype)'''

new = '''    @classmethod
    def gated_delta_net_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        if os.getenv("LOWRANK_GDN") == "1":
            return (state_dtype, state_dtype, torch.int32)
        return (state_dtype, state_dtype)'''

assert old in src, "dtype anchor missing"
src = src.replace(old, new, 1)

# 2. shape: UV + slot
old = '''        temporal_state_shape = (
            divide(num_v_heads, tp_world_size),
            head_k_dim,
            head_v_dim,
        )
        return conv_state_shape, temporal_state_shape'''

new = '''        if os.getenv("LOWRANK_GDN") == "1":
            temporal_state_shape = (
                divide(num_v_heads, tp_world_size),
                head_k_dim + head_v_dim,
                16,
            )
            return conv_state_shape, temporal_state_shape, (1,)
        temporal_state_shape = (
            divide(num_v_heads, tp_world_size),
            head_k_dim,
            head_v_dim,
        )
        return conv_state_shape, temporal_state_shape'''

assert old in src, "shape anchor missing"
src = src.replace(old, new, 1)

# 3. copy func: 3 buffers in lowrank mode
old = '''    def gated_delta_net_state_copy_func(cls):
        return (get_conv_copy_spec, get_temporal_copy_spec)'''

new = '''    def gated_delta_net_state_copy_func(cls):
        if os.getenv("LOWRANK_GDN") == "1":
            return (get_conv_copy_spec, get_temporal_copy_spec, get_temporal_copy_spec)
        return (get_conv_copy_spec, get_temporal_copy_spec)'''

assert old in src, "copy anchor missing"
src = src.replace(old, new, 1)

open(p + ".lr", "w").write(src)
print("patched", p)
