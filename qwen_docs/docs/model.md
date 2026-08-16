# 模型与适配

## 模型

| 项 | 值 |
|---|---|
| 模型 | `Qwen3.8-27B-INT8-W8A8-Dynamic` |
| 量化工具链 | Aoripus-KL3-XLine v2.2-Dev-Nightly |
| 量化模式 | `kl3-compressed-xline`（W8A8 INT8） |
| 目录名 | `models/Qwen3.8-27B-W8A8-INT8-Dynamic`（~30GB，含同名 .zip） |
| 基座架构 | Qwen3.5（`model_type=qwen3_5`） |
| 上下文 | 原生 `max_position_embeddings: 262144`（无 rope_scaling） |

!!! warning "模型路径"
    简写 `models/Qwen3.8` 不存在 → vllm 走 snapshot_download 报 HFValidationError。
    必须用真实目录名 `models/Qwen3.8-27B-W8A8-INT8-Dynamic`。

## W8A8 量化格式

- 权重：**I8 channel 静态对称**（per-channel）
- 激活：**I8 token 动态对称**（per-token）
- 无 zero-point
- `kv_cache_scheme: null`（KV cache 不量化）
- `quantization_status: compressed`
- 量化模式：`kl3-compressed-xline`（Aoripus-KL3-XLine v2.2-Dev-Nightly 导出）

格式指纹已对照昆仑芯 AIAK 输出验证 **11 项 OK**。

### ignore 列表（不参与量化的层）

全部 linear_attn 层（含 norm/in_proj）、`lm_head`、`mtp.*`、`embed_tokens`。
权重 map 含 `model-mtp.safetensors`（MTP 头，加载时被跳过）。

## config.json 权威字段（2026-08-16 复核）

| 字段 | 值 |
|---|---|
| dtype | `float16`（顶层 + text + vision 三处） |
| mamba_ssm_dtype | `float32` |
| max_position_embeddings | `262144`（原生，无 rope_scaling） |
| quantization_status | `compressed` |
| kv_cache_scheme | `null` |
| 导出工具 | transformers 5.13.1 |

## 架构：Qwen3.5 基座（Gated DeltaNet 混合注意力）

Qwen3.8 的 `model_type=qwen3_5` 命中 vllm-kunlun 的 `qwen3_5` 实现：
`Qwen3_5GatedDeltaNet` + `gdn_attention_core` 算子。

| 项 | 说明 |
|---|---|
| 总层数 | 64 层 |
| linear_attn | 48 层（Gated DeltaNet，**无 KV cache**） |
| full_attention | 16 层（**唯一真 KV cache**，64KB/token） |
| KV cache 总量 | 256K 上下文仅 ~16.8GB → 单卡可容 3 个 256K 并发（util 0.9） |
| MTP | 权重被 `skip_prefixes=["mtp."]` 跳过（不做投机解码加速） |

!!! tip "为什么上下文成本低"
    普通 27B Dense 模型 256K 上下文 KV cache 动辄上百 GB；Qwen3.8 只有 1/4 的层
    是全注意力，KV 64KB/token——**256K 仅 ~16.8GB**，这是它能上原生 262144 的关键。

## dtype 与量化的关系（常见误解）

- `--dtype float16` = **计算域精度**（= config 的 dtype 字段）
- INT8 是**权重存储格式**：GEMM 内 INT8×INT8→INT32 累加后反量化回 float16
- LayerNorm / softmax 等非量化层在 float16 域
- 两者不冲突：官方 Qwen3-Coder-480B W8A8 教程同款（config bf16 但启动 float16）
- developer_guide 警告 **avoid bfloat16 due to double VRAM bug**
- 「P800 不支持 FP16」是早前误记：P800 有 xblas `fc_cdnn_infer<float16>` kernel
