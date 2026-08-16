# 启动参数

权威参数（2026-08-16 定稿，03 验证通过）：

| 参数 | 值 | 依据 |
|---|---|---|
| `tensor_parallel_size` | 8 | TP=8 单实例（用户最终决策） |
| `dtype` | `float16` | = config.dtype，计算域精度 |
| `quantization` | `kl3-compressed-xline` | W8A8 INT8 格式（Aoripus-KL3-XLine 导出） |
| `max_model_len` | `262144` | 原生，无 YaRN（业务 ≤256K） |
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

- 默认 0.9 会把 KV cache 预分配顶满 **86.24GiB/卡**（1,331,824 tokens × 64KB），
  GDN SSM state 4.69GiB @256 并发一分配即爆 → short/long 两组 8 卡全 OOM 实证
- 0.75 → KV ~72GiB，留 ~24GiB 给 SSM state + activation
- 03 加载验证（无并发压力）用 0.9 亦可通过

## 关键参数原理

### `--dtype float16`（勿改 bfloat16）

- float16 = 计算域精度；INT8 只是权重存储格式（GEMM 内反量化回 float16）
- 官方 Qwen3-Coder-480B W8A8 教程同款（config bf16 但启动 float16）
- developer_guide 警告：**avoid bfloat16 due to double VRAM bug**
- P800 有 xblas `fc_cdnn_infer<float16>` kernel（「P800 不支持 FP16」是误记）

### `--max-model-len 262144`（原生，无 YaRN）

- 业务确认上下文需求 ≤256K → 不做 YaRN 1M 扩展
- config 原生 `max_position_embeddings: 262144`，无 rope_scaling
- 规避 mrope + partial_rotary 与 YaRN 叠加的兼容风险

### `--mamba-ssm-cache-dtype float16`（PR 408，必需）

- Qwen3.5 的 Gated DeltaNet（GDN）kernel 不支持混合精度
- **不加必挂**——官方 PR 408（Qwen3.5 修复）确认

### `--quantization kl3-compressed-xline`

- 模型是 kl3-compressed-xline 格式（Aoripus-KL3-XLine v2.2-Dev-Nightly 导出的
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
```

## 不要参考的东西

- **vllm 0.25.1 自带 `vllm/platforms/xpu.py` 是 Intel 的**，勿参考（昆仑芯平台在
  `vllm_kunlun/platforms/kunlun.py`）
- `torch/xpu/__init__.py` 是 Intel stub，勿用——一切 device/张量/dist 写 `cuda`
