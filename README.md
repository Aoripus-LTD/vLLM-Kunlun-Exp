# Qwen3.8-27B-INT8-W8A8-Dynamic on Kunlun P800

> 单机 8×96G 昆仑芯 P800-OAM，通过 vllm-kunlun 部署 **Qwen3.8-27B-INT8-W8A8-Dynamic**，
> 原生 262144（256K）上下文，**TP=8 单实例**。
>
> 当前状态（2026-08-16）：**03 加载验证 + 04 吞吐压测全通过** —— TP=8 加载 / 短生成 /
> 8K 长上下文召回 PASS；short 组 256 并发 **3201.71 output tok/s**（稳态峰值 4851）。

---

## 项目概览

本仓库是 [vLLM-Kunlun](https://github.com/baidu/vLLM-Kunlun) 的分支（`qwen38-dev`），目标是
在昆仑芯 XPU 上部署 **Qwen3.8-27B**（基座架构 Qwen3.5，`model_type=qwen3_5`）的
W8A8 INT8 动态量化模型。

| 项 | 说明 |
|---|---|
| 硬件 | 单机 8×P800-OAM（96GB HBM3 × 8，2.4 TB/s/卡） |
| 模型 | `Qwen3.8-27B-INT8-W8A8-Dynamic`（量化模式 `kl3-compressed-xline`） |
| 量化工具链 | Aoripus-KL3-XLine v2.2-Dev-Nightly |
| 上下文 | **原生 262144**（业务需求 ≤256K，不做 YaRN 1M 扩展） |
| 部署形态 | **TP=8 单实例**（最终决策，2026-08-16） |
| 计算域 | float16（`--dtype float16`，P800 有 xblas float16 kernel；INT8 为权重存储格式） |

### 环境版本（官方验证组合，勿随意变动）

| 组件 | 版本 | 说明 |
|---|---|---|
| torch | 2.5.1（xpytorch，CUDA 兼容模式） | `torch.xpu` 不可用，一切 device 写 `cuda` |
| kunlun_ops | 0.1.58 | 官方公开最新版（2026-02-27） |
| vllm | 0.15.1 | PyPI 直装 |
| vllm-kunlun | 0.15.1.dev0（commit 4885de2） | 源码编译，`_kunlun` 已构建 |
| transformers | 5.2.0 | 5.5.3 缺 `max_pixels` 不可用 |
| triton / cocopod / xspeedgate_ops | torch25 配套 | 无变动 |

> **版本对应铁律**：kunlun_ops ↔ vllm-kunlun 源码版本必须对应。0.1.58 只匹配 0.15.1.dev0
> 时代源码；0.25.1-dev（2fda97b）与 0.1.58 接口鸿沟系统性（causal_conv1d 关键字参数 +
> 11 个缺失算子 + 3 个 KW_MISMATCH），曾整体回退（见 [故障排查](./qwen_docs/docs/troubleshooting.md)）。

---

## 快速开始

以下流程均为**已在生产服务器实际跑通**的步骤（docker 容器 `qwen38-p800` 内）。

### 1. 创建容器（8 卡全映射 + 数据路径 /home/newdata）

```bash
docker run -itd --name qwen38-p800 \
  --device=/dev/xpu0..7:/dev/xpu0..7 \
  --device=/dev/xpuctrl:/dev/xpuctrl \
  --net=host --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --tmpfs /dev/shm:rw,nosuid,nodev,exec,size=32g \
  -v /home/newdata:/home/newdata \
  -v /usr/local/bin/xpu-smi:/usr/local/bin/xpu-smi \
  -w /home/newdata \
  docker.int.aoripus.com/vllm-kunlun-exp:base /bin/bash
```

- 容器内 python：`/opt/vllm_kunlun/bin/python`（venv，无 pip）→ 装包一律
  `uv pip install --python /opt/vllm_kunlun/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple`
- 一切数据路径放 `/home/newdata`（14T LVM，docker data-root = `/home/docker`）

### 2. 安装环境（按官方验证组合）

```bash
# 驱动插件/算子栈（kunlun_ops 0.1.58 + triton + xspeedgate_ops + cocopod）
# cocopod 需 UV_SKIP_WHEEL_FILENAME_CHECK=1（文件名与内部版本号不符）

# vLLM（PyPI 直装）+ vllm-kunlun（仓库根 setup.py build+install）
uv pip install vllm==0.15.1 --force-reinstall --no-deps
cd /home/newdata/vLLM-Kunlun-0.25.1-dev && python setup.py build && python setup.py install

# transformers（必须是 5.2.0）
uv pip install transformers==5.2.0 --no-deps --force-reinstall

# 补丁（vllm-kunlun 自带，对 vllm 0.15.x 写 → 11 applied / 0 failed）
python vllm_kunlun/patches/patch_torch251.py
cp vllm_kunlun/patches/eval_frame.py .../site-packages/torch/_dynamo/eval_frame.py
cp vllm_kunlun/quantization/__init__.py .../site-packages/vllm/model_executor/layers/quantization/__init__.py
```

> ⚠️ vllm-kunlun 是 editable 安装形态：`_editable_impl_vllm_kunlun.pth` 加 sys.path 但被
> site-packages 遮蔽 → **实际加载 site-packages/vllm_kunlun**（`__file__` 验证）。同步源码要
> 双份：site-packages + 仓库根；编译必须 cd 仓库根。`_kunlun` 扩展在包根
> `_kunlun.cpython-310-x86_64-linux-gnu.so`（13.5MB，不是 0.25.1 的 `_C/` 子目录）。

### 3. 环境变量（启动前 source）

```bash
source /home/newdata/vLLM-Kunlun-0.25.1-dev/setup_env.sh
# XPU_VISIBLE_DEVICES=0-7  XFT_USE_FAST_SWIGLU=1  XMLIR_CUDNN_ENABLED=1
# XPU_USE_DEFAULT_CTX=1  XMLIR_FORCE_USE_XPU_GRAPH=1  VLLM_HOST_IP=$(hostname -i)
# VLLM_USE_V1=1  USE_ORI_ROPE=1（Qwen3 融合大算子开关）
```

### 4. 加载验证（03，权威参数）

```python
# 03_vllm_load.py 核心参数（qwen38-deploy/ 提供）
LLM(model="/home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic",
    tensor_parallel_size=8,          # TP=8 固定
    dtype='float16',                 # = config.dtype，计算域精度
    quantization='kl3-compressed-xline',
    max_model_len=262144,            # 原生，无 YaRN
    mamba_ssm_cache_dtype='float16', # PR 408 必需（Qwen3.5 GDN kernel 不支持混合精度）
    enable_chunked_prefill=True,
    gpu_memory_utilization=0.9)    # 03 加载验证 0.9 可过；04 压测 256 并发必须 0.75（OOM 修复）
```

验证结果（2026-08-16 实测）：

```
[OK] 模型加载成功  (TP=8, kl3-compressed-xline, float16, max_len=262144)
[短生成]  你好，我是通义千问，由阿里巴巴通义实验室研发的大语言模型。
[长上下文] 模型回答: 200 TOPS        # 8000 token 前的事实，成功召回
[PASS] 召回结果: 包含正确数值 200
```

> ⚠️ 模型目录真实名为 `models/Qwen3.8-27B-W8A8-INT8-Dynamic`（~30GB 含同名 .zip），
> 简写 `models/Qwen3.8` 不存在 → vllm 会走 snapshot_download 报 HFValidationError。

---

## 模型支持

### Qwen3.8-27B-INT8-W8A8-Dynamic（kl3-compressed-xline）

- 量化工具链：Aoripus-KL3-XLine v2.2-Dev-Nightly
- 权重：I8 channel 静态对称；激活：I8 token 动态对称；无 zero-point
- config.json 权威字段：`dtype: float16`（顶层+text+vision 三处）、
  `mamba_ssm_dtype: float32`、`max_position_embeddings: 262144`（原生，无 rope_scaling）、
  `quantization_status: compressed`、`kv_cache_scheme: null`
- `ignore` 列表排除量化：全部 linear_attn 层（含 norm/in_proj）、lm_head、mtp.*、embed_tokens

### 架构（Qwen3.5 基座）

| 项 | 说明 |
|---|---|
| model_type | `qwen3_5` → vllm-kunlun `qwen3_5` 实现（`Qwen3_5GatedDeltaNet` + `gdn_attention_core` 算子） |
| 64 层 | 48 linear_attn（Gated DeltaNet）+ 16 full_attention |
| KV cache | 仅 16 层全注意力产生真 KV cache（64KB/token）→ 256K 仅 ~16.8GB，单卡可容 3 并发 |
| MTP | 权重被 `skip_prefixes=["mtp."]` 跳过（不做投机解码） |

---

## 性能

### XCCL 实测（01_env_check，2026-08-16）

- backend=nccl → 昆仑芯 ProcessGroupXCCL（底层 libbkcl.so），8 设备
- allreduce 2.6MB 单轮 **0.13ms → 19.3 GB/s**（每 token 通信 8.32ms = 64 层 × 0.13ms）
- TP=8 单流理论 ~9.7ms/token（权重读 1.4ms + 通信 8.32ms）

### 官方千帆表（TP=8 单实例口径，@256 并发）

| 模型 | tok/s |
|---|---|
| R1-671B INT8 | 2437 |
| Distill-70B | 4185 |
| Distill-32B | 10328 |
| Distill-14B | 18296 |

### 带宽预算（27B W8A8，权重 ~31GB = I8 24G + F16 7G）

- 单卡单流 decode：31GB / 2.4TB/s ≈ 12.9ms/token（~77 tok/s 带宽上限）
- TP=8 单流：每卡读 3.4GB ≈ 1.4ms + 通信 → 只要 XCCL allreduce ≥15GB/s，TP=8 峰值更高
- 吞吐目标：单流 ≥30 tok/s；几千并发下 5000+ tok/s（官方 R1-671B INT8 口径）

### 部署形态决策（2026-08-16 定稿）

**只跑 TP=8 单实例**（取消双形态对比、取消单卡多实例）。依据：
1. 官方千帆表 = TP=8 单实例口径（R1-671B ≈671GB 必须 TP=8，整表统一测法）→ Dense
   allreduce 量小，XCCL 扛得住（19.3GB/s 实测佐证）
2. 实测「单卡 > 多卡 MoE」是 MoE all-to-all 通信烂（token 搬运量巨大），与 Dense 不矛盾

---

## 部署资产（qwen38-deploy/）

| 资产 | 用途 |
|---|---|
| `01_env_check_v3.sh` + `xccl_bench.py` | 环境自检 + XCCL 带宽实测（19.3 GB/s 实测通过） |
| `03_vllm_load.py` | 加载验证 + 短/长上下文召回（权威参数定稿） |
| `04_throughput_bench.py` | 吞吐压测（short 512/512、long 32768/256，仅 TP=8） |
| `rollback_4885de2_part1/3.sh` | 回退官方验证组合（可重放） |
| `fix_tf_5_2_0.sh` | transformers 5.5.3→5.2.0 修复 |
| `ssh_run.py` | paramiko SSH 执行/上传（连接参数全走环境变量） |
| `hash_verify.sh` / `sync_vllm_kunlun.sh` | 容器↔本机源码双向同步（md5 比对） |

完整部署手册见 [qwen_docs/ 文档站](./qwen_docs/)。

---

## 故障排查

详见 [qwen_docs/docs/troubleshooting.md](./qwen_docs/docs/troubleshooting.md)，摘要：

| # | 问题 | 结论 |
|---|---|---|
| 1-4 | 容器/环境 | 8 卡映射、uv 装包、清华源、平台探测 |
| 5-8 | torch 2.5.1 兼容 | 2fda97b 时代补丁链；回退后不再需要（vllm 0.15.1 原生支持） |
| 9 | kunlun_ops 0.1.58 与 2fda97b 源码鸿沟 | 官方无新版 → **整体回退 4885de2 官方验证组合** |
| 10 | transformers 5.5.3 缺 `max_pixels` | 降级 5.2.0（`qwen2_vl.py:918` AttributeError） |
| 11 | 采样器 exponential_ CPU fallback | decode 卡死根因 → 设备采样 GPU 化（commit b311b51） |

---

## 文档站

[`qwen_docs/`](./qwen_docs/) 为 MkDocs Material 中文文档站，`cd qwen_docs && mkdocs serve` 本地预览。

## License

Apache License 2.0，见 [LICENSE](./LICENSE)。
