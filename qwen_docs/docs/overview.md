# 项目概览

## 项目定位

本仓库为 [vLLM-Kunlun](https://github.com/baidu/vLLM-Kunlun) 的分支（`qwen38-dev`），
面向昆仑芯 XPU 部署 **Qwen3.8-27B-INT8-W8A8-Dynamic**（基座架构 Qwen3.5，
`model_type=qwen3_5`），采用 Aoripus-KL3-XLine v2.2-Dev-Nightly 工具链完成
W8A8 INT8 动态量化导出。

目标指标：

- 单流吞吐 **≥ 30 tok/s**
- 数千并发下 **5000+ tok/s**（官方 R1-671B INT8 口径）
- 业务上下文需求 **≤ 256K**，采用原生 262144，不启用 YaRN 1M 扩展

## 硬件规格（P800-OAM）

| 项 | 规格 |
|---|---|
| 显存 | 96GB HBM3，带宽 **2.4 TB/s** |
| FP16 峰值 | 345 TFLOPS（120W 低功耗模式 128 TFLOPS） |
| INT8 | 文档未公开 TOPS |
| 互联 | XCCL（昆仑芯自研），支持 IB/RoCE |
| TDP | 400W |
| 制程/架构 | 7nm / XPU-P（第三代） |

## 软件版本（官方验证组合，请保持固定）

| 组件 | 版本 | 说明 |
|---|---|---|
| OS | Ubuntu（docker 容器内） | 镜像 `docker.int.aoripus.com/vllm-kunlun-exp:base` |
| python | 3.10.19（`/opt/vllm_kunlun/bin/python`） | venv，无 pip，包安装统一使用 uv |
| torch | 2.5.1（xpytorch，CUDA 兼容模式） | `torch.xpu` 不可用，设备相关代码统一使用 `cuda` |
| kunlun_ops | 0.1.58 | 官方公开最新版（2026-02-27） |
| triton | 3.0.0+b2cde523 | torch25 配套 |
| xspeedgate_ops | 0.0.0+torch25 | 驱动插件 |
| cocopod | 0.0.0+torch25 | 安装需 `UV_SKIP_WHEEL_FILENAME_CHECK=1` |
| vllm | 0.15.1 | PyPI 直装（`--force-reinstall --no-deps`） |
| vllm-kunlun | 0.15.1.dev0（commit 4885de2） | 源码编译，`_kunlun` 已构建 |
| transformers | 5.2.0 | 5.5.3 缺失 `max_pixels` API，不可用 |

!!! warning "版本兼容性要求"
    kunlun_ops 与 vllm-kunlun 源码版本必须对应。kunlun_ops 0.1.58 仅匹配
    0.15.1.dev0 时代源码；0.25.1-dev（2fda97b）与 0.1.58 存在系统性接口不兼容
    （causal_conv1d 关键字参数、11 个缺失算子、3 个 KW_MISMATCH），已整体回退至
    官方验证组合。详见 [故障排查](troubleshooting.md)。

## 关键认知

### 昆仑芯 torch 为 CUDA 兼容模式

- `torch/xpu/__init__.py` 为上游 Intel stub（`_is_compiled()` 恒为 False）
- 真实后端为 **torch.cuda**：`device="cuda:0"` 可创建张量、显存 103.1GB、
  `torch.cuda.device_count()=8`
- `torch.xpu.device_count()` 返回 0 不具备参考意义（lru_cache 缓存 0，驱动正常）；
  设备数量应以 `torch_xmlir._XMLIRC._xpu_get_devices_number()` 为准，返回 8
- 所有 device/张量/分布式代码使用 `cuda`，不使用 `xpu`

### vllm-kunlun 平台层

`KunlunPlatform`（vllm_kunlun/platforms/kunlun.py）：

| 字段 | 值 |
|---|---|
| device_name | `cuda` |
| dispatch_key | `CUDA` |
| dist_backend | `nccl`（对应昆仑芯 ProcessGroupXCCL，底层 libbkcl.so） |
| is_cuda() | False（is_cuda_alike 判定使用） |

### 磁盘格局

- `/home` 为 14T LVM（系统盘仅 `/` 分区 1T）
- docker data-root 为 `/home/docker`（位于 14T 分区）
- 容器数据统一置于 **`/home/newdata`**（模型/源码/驱动/日志，容器 `-v` 挂载同路径）
