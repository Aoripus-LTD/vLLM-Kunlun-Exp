# 启动参数

权威参数（2026-08-16 定稿，03 验证通过）：

| 参数 | 值 | 依据 |
|---|---|---|
| `tensor_parallel_size` | 8 | TP=8 单实例（最终决策） |
| `dtype` | `float16` | 与 config.dtype 一致，计算域精度 |
| `quantization` | `kl3-compressed-xline` | W8A8 INT8 格式（Aoripus-KL3-XLine 导出） |
| `max_model_len` | `262144` | 原生，无 YaRN（业务需求 ≤256K） |
| `mamba_ssm_cache_dtype` | `float16` | **PR 408 必需**（Qwen3.5 GDN kernel 不支持混合精度） |
| `enable_chunked_prefill` | True | 长上下文 prefill 切块 |
| `gpu_memory_utilization` | 0.75 | **04 压测权威值**（OOM 修复，见下） |

```python
LLM(model="/home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic",
    tensor_parallel_size=8,
    dtype='float16',
    quantization='kl3-compressed-xline',
    max_model_len=262144,
    mamba_ssm_cache_dtype='float16',
    enable_chunked_prefill=True,
    gpu_memory_utilization=0.75)
```

### `--gpu-memory-utilization 0.75`（OOM 修复，04 压测权威值）

- 默认 0.9 会将 KV cache 预分配占满 **86.24GiB/卡**（1,331,824 tokens × 64KB），
  GDN SSM state 4.69GiB 在 256 并发下分配即触发 OOM，short/long 两组 8 卡全部
  复现该问题
- 0.75 时 KV 约 72GiB，保留约 24GiB 给 SSM state + activation
- 03 加载验证（无并发压力）使用 0.9 亦可通过

## 关键参数原理

### `--dtype float16`（不要改为 bfloat16）

- float16 为计算域精度；INT8 仅为权重存储格式（GEMM 内反量化回 float16）
- 官方 Qwen3-Coder-480B W8A8 教程同款配置（config bf16，启动 float16）
- developer_guide 警告：**avoid bfloat16 due to double VRAM bug**
- P800 提供 xblas `fc_cdnn_infer<float16>` kernel（「P800 不支持 FP16」为早期记录有误）

### `--max-model-len 262144`（原生，无 YaRN）

- 业务确认上下文需求 ≤256K，不启用 YaRN 1M 扩展
- config 原生 `max_position_embeddings: 262144`，无 rope_scaling
- 规避 mrope + partial_rotary 与 YaRN 叠加的兼容风险

### `--mamba-ssm-cache-dtype float16`（PR 408，必需）

- Qwen3.5 的 Gated DeltaNet（GDN）kernel 不支持混合精度
- 缺少该参数将导致启动失败——官方 PR 408（Qwen3.5 修复）确认

### `--quantization kl3-compressed-xline`

- 模型为 kl3-compressed-xline 格式（Aoripus-KL3-XLine v2.2-Dev-Nightly 导出的
  W8A8 INT8 动态量化），必须显式指定

## 环境变量（启动前 source setup_env.sh）

```text
XPU_VISIBLE_DEVICES=0-7              # 8 卡全可见
XFT_USE_FAST_SWIGLU=1                # 快速 SwiGLU
XMLIR_CUDNN_ENABLED=1                # cudnn 使能
XPU_USE_DEFAULT_CTX=1                # 默认上下文
XMLIR_FORCE_USE_XPU_GRAPH=1          # 强制 XPU Graph
VLLM_HOST_IP=$(hostname -i)          # 分布式通信 IP
VLLM_USE_V1=1                        # V1 引擎
USE_ORI_ROPE=1                       # Qwen3 融合大算子开关
XMLIR_DYNAMO_WORKAROUND=1            # xmlir 1.0.0.1 必需（torch.compile 兼容）
```

> `XMLIR_DYNAMO_WORKAROUND=1` 为 xmlir 1.0.0.1 升级后新增：其 `torch_xmlir/nn/linear.py`
> 的 hydra linear 路径使用 `make_tensors_stateful` contextmanager，torch.compile 会报
> Unsupported；开启该开关后 linear 走 `torch.ops._dynamo_workaround.linear` 才可编译。

## MTP 投机解码启动

```bash
vllm serve /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic \
  --speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}' \
  ...（其余参数同 Dense）
```

- 需要 kunlun_ops **0.1.122**（0.1.58 的 spec conv kernel 会 illegal memory access）
- MTP 与 mamba prefix-caching 不可共存（vllm 0.15.1 限制）
- 实测见 [性能](performance.md)

## 平台适配注意事项

- vllm 0.25.1 自带的 `vllm/platforms/xpu.py` 为 Intel 实现，不应参考（昆仑芯平台位于
  `vllm_kunlun/platforms/kunlun.py`）
- `torch/xpu/__init__.py` 为 Intel stub，不应使用——所有 device/张量/分布式代码
  使用 `cuda`
