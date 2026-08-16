# 项目概览

## 项目定位

本仓库为 [vLLM-Kunlun](https://github.com/baidu/vLLM-Kunlun) 的分支（`qwen38-dev`），
面向昆仑芯 XPU 部署 **Qwen3.8-27B-INT8-W8A8-Dynamic**（基座架构 Qwen3.5，
`model_type=qwen3_5`）的 W8A8 INT8 动态量化模型（Aoripus-KL3-XLine v2.2-Dev-Nightly
量化导出）。

目标指标：

- 单流吞吐 **≥ 30 tok/s**
- 几千并发下 **5000+ tok/s**（官方 R1-671B INT8 口径）
- 业务上下文需求 **≤ 256K** → 原生 262144，不做 YaRN 1M 扩展

## 硬件规格（P800-OAM，mirrorfrog 技术文档确认）

| 项 | 规格 |
|---|---|
| 显存 | 96GB HBM3，带宽 **2.4 TB/s** |
| FP16 峰值 | 345 TFLOPS（120W 低功耗模式 128 TFLOPS） |
| INT8 | 文档未公开 TOPS |
| 互联 | XCCL（昆仑芯自研），支持 IB/RoCE |
| TDP | 400W |
| 制程/架构 | 7nm / XPU-P（第三代） |

## 软件版本（官方验证组合，勿随意变动）

| 组件 | 版本 | 说明 |
|---|---|---|
| OS | Ubuntu（docker 容器内） | 镜像 `docker.int.aoripus.com/vllm-kunlun-exp:base` |
| python | 3.10.19（`/opt/vllm_kunlun/bin/python`） | venv，无 pip → 一律 uv |
| torch | 2.5.1（xpytorch，CUDA 兼容模式） | `torch.xpu` 不可用，一切 device 写 `cuda` |
| kunlun_ops | 0.1.58 | 官方公开最新版（2026-02-27） |
| triton | 3.0.0+b2cde523 | torch25 配套 |
| xspeedgate_ops | 0.0.0+torch25 | 驱动插件 |
| cocopod | 0.0.0+torch25 | 需 `UV_SKIP_WHEEL_FILENAME_CHECK=1` 安装 |
| vllm | 0.15.1 | PyPI 直装（`--force-reinstall --no-deps`） |
| vllm-kunlun | 0.15.1.dev0（commit 4885de2） | 源码编译，`_kunlun` 已构建 |
| transformers | 5.2.0 | 5.5.3 缺 `max_pixels` 不可用 |

!!! warning "版本对应铁律"
    kunlun_ops ↔ vllm-kunlun 源码版本**必须对应**。`kunlun_ops 0.1.58` 只匹配
    0.15.1.dev0 时代源码；0.25.1-dev（2fda97b）与 0.1.58 接口鸿沟系统性
    （causal_conv1d 关键字参数 + 11 个缺失算子 + 3 个 KW_MISMATCH），曾被迫整体回退。
    详见 [故障排查](troubleshooting.md)。

## 关键认知

### 昆仑芯 torch 是 CUDA 兼容模式

- `torch/xpu/__init__.py` 是上游 Intel stub（`_is_compiled()` 恒 False）
- 真实后端是 **torch.cuda**：`device="cuda:0"` 可建张量、103.1GB 显存、`torch.cuda.device_count()=8`
- `torch.xpu.device_count()=0` 是假象（lru_cache 缓存 0，驱动正常）；真验证用
  `torch_xmlir._XMLIRC._xpu_get_devices_number()` = 8
- **所有 device/张量/dist 代码写 `cuda`，勿写 `xpu`**

### vllm-kunlun 平台层

`KunlunPlatform`（vllm_kunlun/platforms/kunlun.py）：

| 字段 | 值 |
|---|---|
| device_name | `cuda` |
| dispatch_key | `CUDA` |
| dist_backend | `nccl`（→ 昆仑芯 ProcessGroupXCCL，底层 libbkcl.so） |
| is_cuda() | False（is_cuda_alike 判定用） |

### 磁盘格局

- **整个 /home = 14T LVM**（系统盘仅 / 分区 1T）
- docker data-root = `/home/docker`（已在 14T）
- 容器数据一律放 **`/home/newdata`**（模型/源码/驱动/日志全在那，容器 `-v` 挂载同路径）
