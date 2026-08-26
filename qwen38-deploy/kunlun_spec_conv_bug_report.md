# kunlun_ops 0.1.58 causal_conv1d_update spec 分支 bug 报告

> 用途：向 vLLM-Kunlun / kunlun_ops 社区反馈 MTP 投机解码适配中定位到的算子问题。
> 本文档为中性技术报告，可作为 kernel 共建 / 算子规格 / 评审验证方向的技术输入。
>
> **状态（2026-08-26）**：该 bug 已在 kunlun_ops **0.1.122**（2026-04-28 torch25 版）
> 中修复，官方 spec kernel 一次调用即可通过（配合 xmlir 1.0.0.1 20260428 版）。
> 本报告保留作为问题定位记录与社区反馈的技术输入。

## 1. 背景

在昆仑芯 P800 上适配 Qwen3.8-27B 的 **MTP（Multi-Token Prediction）投机解码**，走 vllm-kunlun 的
spec decode 路径（`--speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}'`）。
Qwen3.8 为 `qwen3_5` 架构，含 48 个 `linear_attn` 层（Gated DeltaNet / SSM，无 KV cache）。

## 2. 环境

| 组件 | 版本 |
|---|---|
| torch | 2.5.1（昆仑芯 xpytorch，CUDA 兼容模式） |
| kunlun_ops | 0.1.58 |
| vllm | 0.15.1 |
| vllm-kunlun | 0.15.1.dev0（commit 4885de2） |
| 模型 | Qwen3.8-27B-INT8-W8A8-Dynamic（qwen3_5） |

## 3. 问题现象

启用 MTP 后，服务可 READY、cudagraph capture 通过，但第一个 spec decode step 推理即崩：

```
[KERNEL ERROR] 714 (Exception in kernel execution)[.../xBLAS/src/cublas_lt/lt_handle.cpp:22]
RuntimeError: CUDA error: an illegal memory access was encountered
```

逐层二分定位（enforce_eager + checkpoint 打印）确认：数据流（draft token 生成、input_ids 填充）均正确，
崩溃收敛到 `kunlun_ops.causal_conv1d_update` 的 **spec 分支**（传入 `num_accepted_tokens` 非 None 时）。

## 4. 最小复现

```python
import torch
import kunlun_ops

dev = "cuda"
f16 = torch.float16
dim, width, num_blocks = 1280, 4, 16991

weight = torch.randn(dim, width, dtype=f16, device=dev)
conv_state = torch.zeros(num_blocks, width, dim, dtype=f16, device=dev)
idx = torch.tensor([1], dtype=torch.int32, device=dev)

# --- 1. non-spec 调用（num_accepted_tokens 不传）：正常 ---
x_ns = torch.randn(1, 1, dim, dtype=f16, device=dev)
out_ns = torch.empty_like(x_ns)
kunlun_ops.causal_conv1d_update(
    x_ns, weight, out_ns, conv_state, None, None,
    conv_state_indices_cpu=idx.cpu(),
    conv_state_indices_xpu=idx,
    act="SWISH", state_seq_stride=conv_state.stride(0), is_ncw=False,
)
torch.cuda.synchronize()  # OK

# --- 2. spec 调用（num_accepted_tokens 非 None）：illegal memory access ---
x_sp = torch.randn(1, 2, dim, dtype=f16, device=dev)   # (num_spec_decodes, seqlen=2, dim)
out_sp = torch.empty_like(x_sp)
num_acc = torch.tensor([1], dtype=torch.int32, device=dev)
num_acc_cpu = torch.tensor([1], dtype=torch.int32)
kunlun_ops.causal_conv1d_update(
    x_sp, weight, out_sp, conv_state, None, None,
    conv_state_indices_cpu=idx.cpu(),
    conv_state_indices_xpu=idx,
    num_accepted_tokens_cpu=num_acc_cpu,
    num_accepted_tokens_xpu=num_acc,
    act="SWISH", state_seq_stride=conv_state.stride(0), is_ncw=False,
)
torch.cuda.synchronize()  # RuntimeError: CUDA error: an illegal memory access was encountered
```

关键参数：`x=(1, seqlen=2, dim)`、`conv_state_indices=[1]`、`num_accepted_tokens=[1]`。
即「1 个 spec 请求，seq_len=2（1 base + 1 draft），接受 1 个 token」的典型 MTP num_spec=1 场景。

补充验证：dtype 组合（x/weight/conv_state 全 float16 或全 float32，`bias=None`）均复现 spec 分支崩溃；
non-spec 分支在任何 dtype 组合下均正常。可排除 dtype / bias 传参问题。

## 5. 上游对照

vllm 上游 `causal_conv1d_update` 的 spec 路径是 **Triton kernel**（`_causal_conv1d_update_kernel`），
其 spec 语义实现完整：

- spec 时有效 state_len 扩展：`state_len = width - 1 + (seqlen - 1)`
- `conv_state_token_offset = num_accepted_tokens - 1`（按接受数做 conv state 偏移修正）
- `num_accepted_tokens=0` 时 clamp 防负索引

相关上游 PR（GDN/Mamba spec decode state 正确性系列）：

- vllm#40738 — Fix GDN conv + SSM state corruption with ngram spec decode
- vllm#51508 — Skip GDN/KDA recurrent state updates for stale zero-accept spec rows
- vllm#45100 — Avoid racy accepted counts in async spec decode

对照来看，kunlun_ops 0.1.58 的 `causal_conv1d_update` 虽然 **声明了** `num_accepted_tokens_cpu/xpu`
参数（docstring 有），但 spec 分支内核疑似未实现 / 未对齐上述语义，导致越界。

## 6. 期望支持（三选一均可）

1. **内核修复**：修复 kunlun_ops `causal_conv1d_update` 在 `num_accepted_tokens` 非 None 时的越界，
   对齐上游 Triton kernel 的 conv_state_token_offset 语义。
2. **调用规格澄清**：若 spec 语义的调用方式 / 参数约定与我们的用法不同（如 x 的维度顺序、
   num_accepted_tokens 的取值约定、state 布局约定），请提供正确调用规格文档。
3. **版本建议**：是否已有**兼容 torch 2.5.x** 的新版本 kunlun_ops 支持 spec conv
   （当前最新 122 依赖 torch 2.9，与我们的生产组合 torch 2.5.1 不兼容，无法升级）。

## 7. 附：我们已验证的 workaround（供参考，非生产可用）

用 non-spec kernel（dense 生产已验证正常）分两步模拟 spec 语义：
`base token 推进 state → 快照 → draft token 推进 state → 回滚 draft 的推进`。

结果：单流能正确推理（draft 接受率约 75%），证明 spec 语义本身可被 non-spec kernel 拼出；
但存在两个问题——单流吞吐慢约 10 倍（advanced-indexing 快照/回滚开销 + spec 路径额外开销）、
多并发下接受率崩塌（疑似 base/draft 共用 state block 与 spec 真实 state 分配语义不符）。

**结论：Python 层 workaround 可作正确性验证，但不具备生产可用性；根治需 kunlun_ops 内核层面支持 spec conv。**
