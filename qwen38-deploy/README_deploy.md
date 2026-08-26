# Qwen3.8-27B-INT8-W8A8-Dynamic 昆仑芯 P800 部署手册

> 本文为内部部署手册，对外口径以仓库根 README.md 与 qwen_docs/ 文档站为准。

**目标**：单机 8×96G 昆仑芯 P800-OAM，docker 容器 + vllm-kunlun 部署
**Qwen3.8-27B-INT8-W8A8-Dynamic**（Aoripus-KL3-XLine v2.2-Dev-Nightly 量化导出，
量化模式 `kl3-compressed-xline`），原生 **262144（256K）上下文**（不启用 YaRN），
**TP=8 单实例**。

**状态（2026-08-16）**：03 加载验证与 04 吞吐压测全部通过，参数定稿，详见「实测里程碑」。

---

## 环境组合（2026-08-26 更新）

| 组件 | 版本 | 说明 |
|---|---|---|
| torch | 2.5.1 | 昆仑芯 xpytorch，CUDA 兼容模式（设备相关代码使用 cuda，不使用 xpu） |
| torch_xmlir | xmlir 1.0.0.1（2026-04-22 build） | 来自 20260428/torch25 xpytorch `.run`；**0409 版不兼容 0.1.122** |
| kunlun_ops | 0.1.122 | 2026-04-28 torch25 版，MTP spec conv kernel 已修复 |
| vllm | 0.15.1 | PyPI `--force-reinstall --no-deps` |
| vllm-kunlun | 0.15.1.dev0（4885de2 + MTP patches） | 仓库根 setup.py build+install |
| transformers | 5.2.0 | 5.5.3 缺失 max_pixels API（需降级） |
| triton / xspeedgate_ops / cocopod | 3.0.0 / 1.5.0+torch25 / +torch25 | cocopod 需 `UV_SKIP_WHEEL_FILENAME_CHECK=1` |

版本兼容性要求：**驱动插件版本与 vLLM 版本必须对应**，修改任一组件前先核对。
升级要点：

1. kunlun_ops 0.1.122 必须配 xmlir 1.0.0.1 的 **20260428 版**（更早的 0409 版
   缺 31 参 `xfa::gated_delta_net`，0.1.122 import 报 undefined symbol）
2. 0.1.122 的 Python 扩展 `xpu_kunlun_ops` / `xpu_flash_ops` 在 whl 根目录
   （site-packages 顶层），安装时别漏
3. xmlir 升级后启动必须 `export XMLIR_DYNAMO_WORKAROUND=1`（torch.compile
   在 `make_tensors_stateful` 上 Unsupported；已写入 start_serve.sh /
   start_serve_mtp.sh）
4. 旧组合（0.1.58 + 0409 xmlir）备份齐全可回退

## 模型与量化

- **模型名**：`Qwen3.8-27B-INT8-W8A8-Dynamic`
- **量化工具链**：Aoripus-KL3-XLine v2.2-Dev-Nightly
- **量化模式**：`kl3-compressed-xline`（W8A8：权重 I8 channel 静态对称 / 激活 I8 token
  动态对称 / 无 zero-point；`kv_cache_scheme: null`，KV cache 不量化）
- **模型目录**：`models/Qwen3.8-27B-W8A8-INT8-Dynamic`（约 30GB，含同名 .zip；
  简写 `models/Qwen3.8` 不存在，vllm 会触发 snapshot_download 并报 HFValidationError）
- config 权威字段：`dtype: float16`、`mamba_ssm_dtype: float32`、
  `max_position_embeddings: 262144`（无 rope_scaling）、`quantization_status: compressed`
- 架构：Qwen3.5 基座（`model_type=qwen3_5`，对应 vllm-kunlun `qwen3_5` 实现）；
  64 层 = 48 linear_attn（Gated DeltaNet，无 KV cache）+ 16 full_attention
  （KV 64KB/token）；MTP 权重由 `skip_prefixes=["mtp."]` 跳过

## 权威启动参数（定稿）

```text
--tensor-parallel-size 8           TP=8 单实例（最终决策）
--dtype float16                    与 config dtype 一致；不要改为 bfloat16（double VRAM bug）
--max-model-len 262144             原生，无 YaRN（业务上下文需求 ≤256K）
--quantization kl3-compressed-xline  W8A8 量化模式
--mamba-ssm-cache-dtype float16    PR 408 必需（Qwen3.5 GDN kernel 不支持混合精度）
--gpu-memory-utilization 0.75      04 压测权威值（0.9 会将 KV 预分配占满 86.24GiB/卡
                                   导致 OOM；03 加载验证无并发压力时 0.9 亦可）
--enable-chunked-prefill           长上下文 prefill 切块
```

环境变量：`source .../vLLM-Kunlun-0.25.1-dev/setup_env.sh`（XPU_VISIBLE_DEVICES=0-7、
XFT_USE_FAST_SWIGLU=1、XMLIR_CUDNN_ENABLED=1、XPU_USE_DEFAULT_CTX=1、
XMLIR_FORCE_USE_XPU_GRAPH=1、VLLM_HOST_IP=$(hostname -i)、USE_ORI_ROPE=1、
**XMLIR_DYNAMO_WORKAROUND=1**——xmlir 1.0.0.1 升级后必需，已写入启动脚本）

## 实测里程碑（2026-08-16）

- **XCCL**：allreduce 2.6MB 单轮 0.13ms，**19.3 GB/s**（8 设备，nccl → ProcessGroupXCCL）
- **03 加载验证**（TP=8）：加载成功 → 短文本生成正常 → 8K 长上下文召回「200 TOPS」PASS
- **04 吞吐压测**：
  - short（512/512 × 256 并发）：**6.25 req/s / 3201.71 output tok/s**
    （稳态峰值 4851 output tok/s，满 batch）
  - long（32768/256 × 16）：**84.02 output tok/s**（total 10839，prefill 主导，约 45s）
- **采样器 exponential_ CPU fallback（decode 停滞根因，已改为设备端采样）**：
  TopKTopPSampler 无 k/p 过滤时 `q.exponential_()` 走 CPU fallback（256 batch ×
  152K vocab 实测 4.89s/step，GPU 0%）——修复：无 generators 时一律走
  flashinfer_sample 设备端采样（k/p=None 时内部补充 top_k=vocab/top_p=1.0）；
  generators 分支改用 uniform_ + (-log) 统计等价的指数噪声。修改位于
  vllm_kunlun/v1/sample/ops/topk_topp_sampler.py（容器双份同步：site-packages +
  仓库根）

## 实测里程碑（2026-08-26，0.1.122 栈 + MTP）

- **算子栈升级**：kunlun_ops 0.1.122 + xmlir 1.0.0.1（20260428 版），官方 MTP
  spec conv kernel 修复确认（0 崩溃），单流 Mean acceptance length 1.83
- **API 层 overall 吞吐**（prompt 172 + output 256）：

  | 形态 | 单流 | 8 并发 | 32 并发 | 256 并发 |
  |---|---|---|---|---|
  | Dense（0.1.58 栈） | 57 | 220 | 665 | 1542 |
  | Dense（0.1.122 栈） | 52 | — | — | **1552** |
  | **MTP（0.1.122 栈）** | **55-62** | 246 | 709 | 1482 |

- **MTP 启动**：`start_serve_mtp.sh`（`--speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}'`）；
  causal_conv1d spec 分支已切换为官方单次 kernel（commit e173c5e）
- **剩余优化点**：MTP iteration 28-38ms（enforce-eager 下 layer0 计时：conv
  0.45ms / recurrent 0.23ms / 输入投影 1.4ms），单流仍有约 2x 优化空间
- **生产形态选择**：高并发负载用 Dense（新栈 1552 tok/s），单流敏感场景用 MTP

## 部署操作

### 1. 创建容器（8 卡全映射）

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
  <基础镜像> /bin/bash
```

操作要求：宿主机不执行任何 vllm 相关命令（仅允许 docker create/start/exec）；
容器内 python 位于 `/opt/vllm_kunlun/bin/python`（venv，无 pip），包安装统一使用
`uv pip install --python /opt/vllm_kunlun/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple`；
pip 使用国内源；数据路径全部位于 `/home/newdata`。

### 2. 安装（官方验证组合）

```bash
uv pip install vllm==0.15.1 --force-reinstall --no-deps
cd <vLLM-Kunlun 源码根>   # 构建文件所在目录
python setup.py build && python setup.py install
uv pip install transformers==5.2.0 --no-deps --force-reinstall
python vllm_kunlun/patches/patch_torch251.py   # 11 applied / 0 failed
cp vllm_kunlun/patches/eval_frame.py /opt/vllm_kunlun/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py
cp vllm_kunlun/quantization/__init__.py /opt/vllm_kunlun/lib/python3.10/site-packages/vllm/model_executor/layers/quantization/__init__.py
```

editable 安装形态说明：import 实际加载 site-packages/vllm_kunlun（`.pth` 被遮蔽），
源码同步需双份；`_kunlun` 扩展位于包根 `_kunlun.cpython-310-x86_64-linux-gnu.so`。

### 3. 验证 + 压测

```bash
python 03_vllm_load.py --model models/Qwen3.8-27B-W8A8-INT8-Dynamic
python 04_throughput_bench.py --model models/Qwen3.8-27B-W8A8-INT8-Dynamic --group short
python 04_throughput_bench.py --model models/Qwen3.8-27B-W8A8-INT8-Dynamic --group long --num-prompts 16
```

注：vllm 0.15.1 的 `benchmark_throughput.py` 为废弃占位，正确入口为
`vllm bench throughput` CLI（04 脚本已封装）。

## 文件清单（qwen38-deploy/）

| 文件 | 用途 | 运行位置 |
|---|---|---|
| `01_env_check_v3.sh` + `xccl_bench.py` | 环境自检 + XCCL allreduce 带宽基准（实测 19.3 GB/s） | 服务器（容器内） |
| `03_vllm_load.py` | 加载验证 + 短/长上下文召回（TP=8 定稿参数） | 服务器（容器内） |
| `04_throughput_bench.py` | 吞吐压测（short/long，`vllm bench throughput` 封装） | 服务器（容器内） |
| `run03.sh` / `run04.sh` + `watch_03_run.py` / `watch04.py` | 03/04 编排 + 日志轮询（SSH 凭据走环境变量） | 服务器（容器内）/ 本机 |
| `ssh_run.py` | paramiko SSH 执行/上传（连接参数全部走环境变量） | 本机 |
| `fetch_docs.py` | 文档站全站爬取器 | 本机 |
| `rollback_4885de2_part1/2/3.sh`、`fix_tf_5_2_0.sh` | 官方验证组合可重放安装/回退脚本 | 服务器（容器内） |
| `verify_rollback.sh`、`probe_rollback.sh`、`probe_tf.sh` | 安装形态/版本验证 | 服务器（容器内） |
| `check_llm_sig.sh`、`check_vllm_params.sh` | LLM 签名 / CacheConfig 字段检查 | 服务器（容器内） |
| `sync_vllm_kunlun.sh`、`hash_verify.sh` | 容器与本机源码双向同步（md5 清单比对） | 本机/服务器 |

## SSH 工具用法（ssh_run.py）

```powershell
# PowerShell（本机）
$env:QWEN38_HOST='服务器地址'
$env:QWEN38_PORT='端口'            # 可选，默认 22
$env:QWEN38_USER='root'            # 可选，默认 root
$env:QWEN38_SSH_PASS='密码'
python ssh_run.py "命令"
```

!!! note "ssh_run.py 注意事项"
    - 连接参数与凭据**绝不入库**，全部从环境变量读取
    - stdout 含 emoji 时需加 `PYTHONIOENCODING=utf-8` 前缀（Windows GBK 下会
      UnicodeEncodeError）
    - **多余位置参数会追加进远程命令**——不要传入多余参数
    - 参数不要使用双引号（PowerShell 会吞引号）——复杂命令写 .sh 上传执行

## 已知问题与修复记录

1. **版本兼容性**：kunlun_ops 0.1.58 仅匹配 0.15.1.dev0 时代源码（更新的 2fda97b
   源码接口系统性不匹配：causal_conv1d 关键字参数 + 缺失算子 + KW_MISMATCH）——
   回退至官方验证组合（issue #387 同款）
2. **transformers 5.5.3 缺失 `max_pixels`**——必须使用 5.2.0（`fix_tf_5_2_0.sh`）
3. **vllm 0.15.1 random 参数注意事项**：`--random-input-len/output-len` 默认
   1024/128 且优先于定长参数——04 脚本直接传 random 版本参数
4. **KV 预算 OOM**：`--gpu-memory-utilization 0.9` 将 KV 预分配占满 86.24GiB/卡，
   GDN SSM state 在 256 并发下分配即触发 OOM——改为 0.75（short/long 两组 8 卡
   全部复现该问题）
5. **采样器 CPU fallback**：见「实测里程碑」修复记录
6. **CRLF 处理**：容器侧源码上传必须使用 `git archive`（LF 级），md5 比对去除 \r
