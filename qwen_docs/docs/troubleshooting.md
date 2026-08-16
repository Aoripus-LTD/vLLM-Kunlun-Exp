# 故障排查

部署全程遇到的主要问题，按时间顺序记录。**修复脚本均在 qwen38-deploy/ 目录，可重放**。

## 问题总览

| # | 问题 | 结论 |
|---|---|---|
| 1 | 容器/环境 | 8 卡映射、uv 装包、清华源、平台探测 |
| 2 | 模型路径简写 | `models/Qwen3.8` 不存在，触发 HFValidationError |
| 3 | torch.xpu 计数误导 | CUDA 兼容模式，设备相关代码统一使用 cuda |
| 4 | torch.accelerator mock | 2fda97b 时代方案（回退后不再需要） |
| 5-8 | torch 2.5.1 兼容补丁链 | 2fda97b 时代补丁（回退后失效，vllm 0.15.1 原生支持） |
| 9 | **kunlun_ops 0.1.58 与 2fda97b 源码不兼容** | **官方无新版，整体回退 4885de2 官方验证组合** |
| 10 | transformers 5.5.3 缺失 `max_pixels` | 降级 5.2.0 |
| 11 | **采样器 exponential_ CPU fallback** | **decode 停滞根因，改为 kunlun_ops 设备端采样**（另附：KV 预算 OOM + random 参数注意事项） |

## 环境与平台（2026-08-15）

1. **容器 8 卡映射**：`--device=/dev/xpu0..7:/dev/xpu0..7` + `--device=/dev/xpuctrl`，
   映射不完整会导致所有分布式操作失败
2. **模型路径**：简写 `models/Qwen3.8` 不存在，vllm 触发 snapshot_download 并立即
   报 HFValidationError。真实目录名为 `models/Qwen3.8-27B-W8A8-INT8-Dynamic`
3. **torch.xpu.device_count()=0 不具备参考意义**：torch 原生模块 lru_cache 缓存 0，
   驱动正常。设备数量以 `torch_xmlir._XMLIRC._xpu_get_devices_number()` 为准，返回 8
4. **昆仑芯 torch251 为 CUDA 兼容模式**：`torch/xpu/__init__.py` 为 Intel stub
   （`_is_compiled()` 恒为 False）；真实后端为 torch.cuda（`device="cuda:0"` 可创建
   张量、显存 103.1GB、device_count=8）。vllm-kunlun 的 KunlunPlatform：
   `device_name="cuda"`、`dispatch_key="CUDA"`、`dist_backend="nccl"`、`is_cuda()=False`。
   所有 device/张量/分布式代码使用 cuda，不使用 xpu

## torch 2.5.1 兼容补丁链（2026-08-16，仅 2fda97b 时代有效）

vllm 0.25.1（PyPI）要求 torch==2.11.0（`torch.accelerator.get_memory_info()` 2.6+
才提供）；昆仑芯 xpytorch 仅有 2.5.1 / 2.9 两代。0.25.1 + torch 2.5.1 组合需要
8 个补丁：

- **Patch2 torch.Size trace**（attention.py:527）
- **pass_manager 平台条件放宽**（不触碰 is_cuda 专属区块）
- **auto_functionalized 导出**（matcher_utils.py + `torch/_higher_order_ops/__init__.py`）
- **fusion passes 关闭**（enable_norm_fusion/enable_act_fusion 增加昆仑芯判定：
  device_name=="cuda" 且 is_cuda_alike()==False 时返回 False）

!!! info "回退后不再需要"
    vllm 0.15.1 **原生支持 torch 2.5.1**，patch_torch251 11/11 全部适用，上述 8 个
    补丁不再需要。此链仅作 2fda97b 时代的历史记录。

## kunlun_ops 0.1.58 与 2fda97b 源码系统性不兼容（回退根因）

### 症状

vllm 0.25.1-dev（2fda97b）源码调用 kunlun_ops 0.1.58 时：

- `causal_conv1d` 关键字参数不匹配（KW_MISMATCH ×3）
- **11 个缺失算子**
- 接口差异具有系统性，无法通过补丁修复

### 排查过程（结论）

kunlun_ops 0.1.58（2026-02-27）为**官方公开最新版**：

- 文档 stable/latest/main 三处均无新版
- docker hub 5 个 tags 均无新版
- base_v0.0.2 镜像经用户确认无新版

0.1.58 仅匹配 **0.15.1.dev0 时代源码**（4885de2）。

### 决策

**整体回退至官方验证组合**（issue #387 同款）：

```text
vllm-kunlun: 2fda97b (0.25.1-dev) → 4885de2 (0.15.1.dev0)
vllm:        0.25.1 → 0.15.1
transformers: 5.5.3 → 5.2.0
torch / kunlun_ops / triton / cocopod: 保持不变
```

重放脚本：`rollback_4885de2_part1.sh` + `rollback_4885de2_part3.sh`
（part2 存在路径问题，已并入 part3；备份位于 `/home/newdata/backup/`，4.8M tar.gz）。

### 回退后验证

- patch_torch251：**11 applied | 0 failed**（针对 vllm 0.15.x）
- `_kunlun` .so 位置：包根 `_kunlun.cpython-310-x86_64-linux-gnu.so`（13.5MB，
  非 0.25.1 的 `_C/` 子目录）
- import vllm 通过

## transformers 5.5.3 缺失 max_pixels API

### 症状

03 首次运行 `Engine core initialization failed`：

```text
AttributeError: 'Qwen2VLImageProcessor' object has no attribute 'max_pixels'
```

### 根因

- 容器内 transformers=5.5.3（0.25.1 时代安装）
- 5.5.3 重构后移除 `Qwen2VLImageProcessor.max_pixels`
- vllm 0.15.1 `qwen2_vl.py:918`（MultiModalBudget）读取该属性，触发 AttributeError

### 修复

```bash
uv pip install transformers==5.2.0 --no-deps --force-reinstall   # fix_tf_5_2_0.sh
```

!!! note "排查经验"
    类属性检查 `hasattr(Qwen2VLImageProcessor, 'max_pixels')` 返回 False，但实例
    属性存在（`__init__` 从 config 设置）——无需深入排查，直接降级验证即可。

## 采样器 exponential_ CPU fallback（decode 停滞根因）

### 症状

04 压测 256 并发 decode 阶段 GPU 占用归零、吞吐停滞于 32 tok/s、进度条无进展；
带 seed 的请求解码直接 AttributeError 崩溃。

### 根因

- `TopKTopPSampler` 无 k/p 过滤（top_k=0/top_p=1.0 归一化后 k/p=None）时走
  `forward_native`，调用 `random_sample` 的 `q.exponential_()`
- 昆仑芯 CUDA 兼容层**未提供 exponential_ 的设备 kernel**，静默回退至 CPU
- 实测：256 batch × 152K vocab 单步 **4.89s，GPU 0%**（probe_sampler.py 证据）

### 修复（commit b311b51）

`vllm_kunlun/v1/sample/ops/topk_topp_sampler.py`（容器 site-packages 与仓库根
双份同步）：

1. **`forward_kunlun` 无 generators 时一律走 `flashinfer_sample` 设备端采样**，
   不再进入 forward_native；k/p=None 时内部补充 top_k=vocab / top_p=1.0
   （数学上等价于无过滤的纯随机采样）
2. **generators 分支**：`xspeedgate_ops.inplace_exponential` 在当前官方组合
   （xspeedgate_ops-0.0.0+torch25）中不存在（AttributeError）——统一改用
   `uniform_ + (-log)` 统计等价的指数噪声（-log(U) ~ Exp(1)，与 FAST_RANDOM_SAMPLE
   同款算法；uniform_ 有设备 kernel，实测 0.01s）

### 效果

short 512/512 @256 并发 output 吞吐由 32 提升至 **3201.71 tok/s（约 72 倍）**，
稳态峰值 4851 tok/s，RUN04_EXIT=0。

### 同期注意事项（04 压测）

- **KV 预算 OOM**：`--gpu-memory-utilization 0.9` 将 KV 预分配占满 86.24GiB/卡，
  GDN SSM state 4.69GiB 在 256 并发下分配即触发 OOM，8 卡全部复现——改用 **0.75**
  （KV 约 72GiB，保留约 24GiB 给 SSM state + activation）
- **random 参数注意事项**：vllm 0.15.1 `--random-input-len/--random-output-len`
  存在默认值（1024/128）且优先于定长参数——04 脚本直接传 random 版本参数
- **benchmark 入口**：v0.15.1 的 benchmark_throughput.py 为废弃占位（exit 1），
  正确入口为 `vllm bench throughput` CLI

## vllm 0.15.1 结构变化注意事项

回退后需注意 vllm 0.15.1 与 0.25.1 的结构差异：

- config 为包（`vllm/config/cache.py` 等）
- `LLM.__init__` 为薄封装（显式参数少，其余参数经 **kwargs 透传 EngineArgs）——
  03 的 `enable_chunked_prefill` 等参数经透传可用（已实测验证）

## 操作经验

- **PowerShell 引号**：`\"` 会被吞掉、单引号内嵌双引号传给远程 bash 会被剥离、
  `|` 管道会被 PowerShell 解析——**应始终写 .sh 文件上传执行**
- **grep/ls 必须带 docker exec**：遗漏时命令会在宿主机执行，报「没有那个文件或
  目录」产生误报
- **CRLF 处理**：本机 `git reset --hard` 在 .gitattributes 生效前执行过，工作树被
  autocrlf 转为 CRLF（git 内仍为 LF）——上传容器源码使用 `git archive`，md5 比对
  去除 `\r`
