# Qwen3.8-27B-INT8-W8A8-Dynamic on Kunlun P800

> 在单机 8×96G 昆仑芯 P800-OAM 上，通过 vllm-kunlun 部署 **Qwen3.8-27B-INT8-W8A8-Dynamic**，
> 原生 **262144（256K）** 上下文，**TP=8 单实例**。

**部署状态（2026 年 8 月 26 日更新）**：加载验证与吞吐压测均已通过——TP=8 加载、
短文本生成、8K 长上下文召回全部正常；short 组 256 并发实测 **3201.71 output tok/s**。
算子栈已升级至 kunlun_ops 0.1.122 + xmlir 1.0.0.1（20260428 版），官方 MTP
spec conv kernel 修复确认，MTP 单流 **55-62 tok/s**（超越同栈 Dense），详见
[性能](./qwen_docs/docs/performance.md)。

---

## 项目概览

本仓库为 [vLLM-Kunlun](https://github.com/baidu/vLLM-Kunlun) 的分支（`qwen38-dev`），
目标是在昆仑芯 XPU 上部署 **Qwen3.8-27B** 的 W8A8 INT8 动态量化模型（基座架构
Qwen3.5，`model_type=qwen3_5`）。

| 项 | 说明 |
|---|---|
| 硬件 | 单机 8×P800-OAM（96GB HBM3 × 8，单卡带宽 2.4 TB/s） |
| 模型 | `Qwen3.8-27B-INT8-W8A8-Dynamic`，量化模式 `kl3-compressed-xline` |
| 量化工具链 | Aoripus-KL3-XLine v2.2-Dev-Nightly |
| 上下文长度 | **原生 262144**，不启用 YaRN 扩展 |
| 部署形态 | **TP=8 单实例**（2026 年 8 月 16 日定稿） |
| 计算域精度 | float16（`--dtype float16`；INT8 为权重存储格式） |

### 环境版本（2026-08-26 更新）

| 组件 | 版本 | 说明 |
|---|---|---|
| torch | 2.5.1（xpytorch，CUDA 兼容模式） | 设备/张量/分布式代码一律使用 `cuda`，不使用 `xpu` |
| torch_xmlir | xmlir 1.0.0.1（2026-04-22 build） | 来自 20260428/torch25 xpytorch 包；升级后需 `XMLIR_DYNAMO_WORKAROUND=1` |
| kunlun_ops | 0.1.122 | 2026-04-28 torch25 版，修复 MTP spec conv kernel |
| vllm | 0.15.1 | PyPI 直装 |
| vllm-kunlun | 0.15.1.dev0（commit 4885de2 + MTP patches） | 源码编译，`_kunlun` 扩展已构建 |
| transformers | 5.2.0 | 5.5.3 缺失 `max_pixels` API，不可用 |
| triton / cocopod / xspeedgate_ops | torch25 配套版本 | 未变更 |

> **版本兼容性要求**：kunlun_ops 与 vllm-kunlun 源码版本必须对应。kunlun_ops 0.1.58
> 仅匹配 0.15.1.dev0 时代源码；0.25.1-dev（2fda97b）与 0.1.58 存在系统性接口不兼容
> （causal_conv1d 关键字参数、11 个缺失算子、3 个 KW_MISMATCH），曾回退至官方验证组合。
> kunlun_ops 0.1.122 必须搭配 xmlir 1.0.0.1 的 **20260428 版**（更早的 0409 版缺少
> 31 参 `xfa::gated_delta_net` 符号，0.1.122 import 会失败）。详见
> [故障排查](./qwen_docs/docs/troubleshooting.md)。

---

## 快速开始

以下流程均在 docker 容器 `qwen38-p800` 内实际验证通过。

### 1. 创建容器（8 卡全映射，数据路径 /home/newdata）

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

- 容器内 python 位于 `/opt/vllm_kunlun/bin/python`（venv，无 pip），包安装统一使用
  `uv pip install --python /opt/vllm_kunlun/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple`
- 全部数据路径置于 `/home/newdata`（14T LVM；docker data-root 为 `/home/docker`）

### 2. 安装环境（官方验证组合）

```bash
# 驱动插件/算子栈（kunlun_ops 0.1.58 + triton + xspeedgate_ops + cocopod）
# cocopod 安装需设置 UV_SKIP_WHEEL_FILENAME_CHECK=1（wheel 文件名与内部版本号不一致）

# vLLM（PyPI 直装）与 vllm-kunlun（仓库根 setup.py build + install）
uv pip install vllm==0.15.1 --force-reinstall --no-deps
cd /home/newdata/vLLM-Kunlun-0.25.1-dev && python setup.py build && python setup.py install

# transformers（必须为 5.2.0）
uv pip install transformers==5.2.0 --no-deps --force-reinstall

# 补丁（vllm-kunlun 自带，针对 vllm 0.15.x：11 applied / 0 failed）
python vllm_kunlun/patches/patch_torch251.py
cp vllm_kunlun/patches/eval_frame.py .../site-packages/torch/_dynamo/eval_frame.py
cp vllm_kunlun/quantization/__init__.py .../site-packages/vllm/model_executor/layers/quantization/__init__.py
```

> **editable 安装说明**：vllm-kunlun 采用 editable 安装形态，`_editable_impl_vllm_kunlun.pth`
> 将仓库根加入 sys.path，但实际加载路径为 site-packages/vllm_kunlun（可用 `__file__`
> 验证）。源码同步需双份（site-packages 与仓库根）；编译必须位于仓库根执行。
> `_kunlun` 扩展位于包根 `_kunlun.cpython-310-x86_64-linux-gnu.so`（13.5MB）。

### 3. 环境变量（启动前 source）

```bash
source /home/newdata/vLLM-Kunlun-0.25.1-dev/setup_env.sh
# XPU_VISIBLE_DEVICES=0-7  XFT_USE_FAST_SWIGLU=1  XMLIR_CUDNN_ENABLED=1
# XPU_USE_DEFAULT_CTX=1  XMLIR_FORCE_USE_XPU_GRAPH=1  VLLM_HOST_IP=$(hostname -i)
# VLLM_USE_V1=1  USE_ORI_ROPE=1（Qwen3 融合大算子开关）
```

### 4. 加载验证

```python
# qwen38-deploy/03_vllm_load.py 核心参数
LLM(model="/home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic",
    tensor_parallel_size=8,          # TP=8 固定
    dtype='float16',                 # 与 config.dtype 一致，计算域精度
    quantization='kl3-compressed-xline',
    max_model_len=262144,            # 原生，不启用 YaRN
    mamba_ssm_cache_dtype='float16', # PR 408 必需（Qwen3.5 GDN kernel 不支持混合精度）
    enable_chunked_prefill=True,
    gpu_memory_utilization=0.9)      # 加载验证 0.9 可通过；压测 256 并发需 0.75
```

验证结果（2026 年 8 月 16 日实测）：

```
[OK] 模型加载成功  (TP=8, kl3-compressed-xline, float16, max_len=262144)
[短生成]  你好，我是通义千问，由阿里巴巴通义实验室研发的大语言模型。
[长上下文] 模型回答: 200 TOPS        # 8000 token 前的事实，成功召回
[PASS] 召回结果: 包含正确数值 200
```

> **模型目录**：真实目录名为 `models/Qwen3.8-27B-W8A8-INT8-Dynamic`（约 30GB，含同名
> .zip）。简写 `models/Qwen3.8` 不存在，vllm 会触发 snapshot_download 并报
> HFValidationError。

---

## 模型支持

### Qwen3.8-27B-INT8-W8A8-Dynamic（kl3-compressed-xline）

- 量化工具链：Aoripus-KL3-XLine v2.2-Dev-Nightly
- 权重 I8 channel 静态对称；激活 I8 token 动态对称；无 zero-point
- config.json 权威字段：`dtype: float16`（顶层 + text + vision 三处）、
  `mamba_ssm_dtype: float32`、`max_position_embeddings: 262144`（原生，无
  rope_scaling）、`quantization_status: compressed`、`kv_cache_scheme: null`
- `ignore` 列表排除的量化层：全部 linear_attn 层（含 norm/in_proj）、`lm_head`、
  `mtp.*`、`embed_tokens`

### 模型架构（Qwen3.5 基座）

| 项 | 说明 |
|---|---|
| model_type | `qwen3_5`，对应 vllm-kunlun `qwen3_5` 实现（`Qwen3_5GatedDeltaNet` + `gdn_attention_core` 算子） |
| 层结构 | 64 层：48 linear_attn（Gated DeltaNet）+ 16 full_attention |
| KV cache | 仅 16 层全注意力产生 KV cache（64KB/token）；256K 上下文约 16.8GB，单卡可容纳 3 个并发请求 |
| MTP | 权重通过 `skip_prefixes=["mtp."]` 跳过（不启用投机解码） |

---

## 性能

### XCCL 实测（2026 年 8 月 16 日）

- backend=nccl，对应昆仑芯 ProcessGroupXCCL（底层 libbkcl.so），8 设备
- allreduce 2.6MB 单轮 **0.13ms，约 19.3 GB/s**（每 token 通信 8.32ms，即 64 层 × 0.13ms）
- TP=8 单流理论时延约 9.7ms/token（权重读取 1.4ms + 通信 8.32ms）

### 官方性能基准（TP=8 单实例，256 并发）

| 模型 | tok/s |
|---|---|
| R1-671B INT8 | 2437 |
| Distill-70B | 4185 |
| Distill-32B | 10328 |
| Distill-14B | 18296 |

### 实测结果（2026 年 8 月 26 日，kunlun_ops 0.1.122 栈）

short 组（input 512 / output 512 × 256 并发）与 MTP 投机解码实测（API 层 overall，
prompt 172 + output 256）：

| 形态 | 单流 overall | 256 并发 overall |
|---|---|---|
| Dense（0.1.58 栈） | 57 tok/s | 1542 tok/s |
| Dense（0.1.122 栈） | 52 tok/s | **1552 tok/s** |
| **MTP（0.1.122 栈）** | **55-62 tok/s** | 1482 tok/s |

- MTP（`num_speculative_tokens=1, method=mtp`）Mean acceptance length **1.83**，
  单流首次反超 Dense（+9%）；高并发下 draft 计算与验证争抢算力，比 Dense 略低
  （-4.5%），生产形态按负载特征选择
- MTP 与 mamba prefix-caching 不可共存（vllm 0.15.1 限制）

### 带宽预算（27B W8A8，权重约 31GB，I8 24G + F16 7G）

- 单卡单流 decode：31GB / 2.4TB/s ≈ 12.9ms/token（约 77 tok/s 带宽上限）
- TP=8 单流：每卡读取 3.4GB ≈ 1.4ms + 通信开销。XCCL allreduce 实测 19.3GB/s，
  高于 15GB/s 临界值，TP=8 峰值吞吐更高
- 吞吐目标：单流 ≥30 tok/s；数千并发下 5000+ tok/s（官方 R1-671B INT8 口径）

### 部署形态决策（2026 年 8 月 16 日定稿）

**仅采用 TP=8 单实例**。依据：

1. 官方性能基准为 TP=8 单实例口径（R1-671B INT8 约 671GB，必须 TP=8），Dense 模型
   allreduce 通信量小，实测 19.3GB/s 可满足
2. 「单卡优于多卡 MoE」的实测结论源于 MoE all-to-all 通信开销（token 搬运量大），
   与 Dense 模型不矛盾

---

## 部署资产（qwen38-deploy/）

| 资产 | 用途 |
|---|---|
| `01_env_check_v3.sh` + `xccl_bench.py` | 环境自检 + XCCL 带宽基准（实测 19.3 GB/s） |
| `03_vllm_load.py` | 加载验证 + 短/长上下文召回 |
| `04_throughput_bench.py` | 吞吐压测（short 512/512、long 32768/256，仅 TP=8） |
| `start_serve_mtp.sh` | MTP 投机解码服务启动（num_speculative_tokens=1） |
| `bench_stream.py` / `bench1.py` | 单流/流式 token 间隔基准 |
| `start_serve_mtp_prof.sh` | MTP + torch profiler 启动（profiling 用） |
| `rollback_4885de2_part1/3.sh` | 官方验证组合回退脚本（可重放） |
| `fix_tf_5_2_0.sh` | transformers 5.5.3 → 5.2.0 修复 |
| `ssh_run.py` | paramiko SSH 执行/上传（连接参数全部走环境变量） |
| `hash_verify.sh` / `sync_vllm_kunlun.sh` | 容器与本机源码双向同步（md5 比对） |

完整部署手册见 [qwen_docs/ 文档站](./qwen_docs/)。

---

## 故障排查

摘要见下，完整记录见 [qwen_docs/docs/troubleshooting.md](./qwen_docs/docs/troubleshooting.md)。

| # | 问题 | 结论 |
|---|---|---|
| 1-4 | 容器/环境 | 8 卡映射、uv 装包、清华源、平台探测 |
| 5-8 | torch 2.5.1 兼容 | 2fda97b 时代补丁链，回退后不再需要（vllm 0.15.1 原生支持） |
| 9 | kunlun_ops 0.1.58 与 2fda97b 源码接口不兼容 | 官方无新版，整体回退 4885de2 官方验证组合 |
| 10 | transformers 5.5.3 缺失 `max_pixels` | 降级 5.2.0（`qwen2_vl.py:918` AttributeError） |
| 11 | 采样器 exponential_ CPU fallback | decode 卡死根因，已改为设备端采样（commit b311b51） |
| 12 | kunlun_ops 0.1.122 import undefined symbol | 需配 xmlir 1.0.0.1 20260428 版 + `XMLIR_DYNAMO_WORKAROUND=1`（详见 [安装](./qwen_docs/docs/installation.md)） |

---

## 文档站

[`qwen_docs/`](./qwen_docs/) 为 MkDocs Material 中文文档站。本地预览：
`cd qwen_docs && mkdocs serve`。

## License

Apache License 2.0，见 [LICENSE](./LICENSE)。
