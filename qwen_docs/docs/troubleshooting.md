# 故障排查

部署全程遇到的主要问题，按时间顺序记录。**修复脚本均在 qwen38-deploy/ 可重放**。

## 问题总览

| # | 问题 | 结论 |
|---|---|---|
| 1 | 容器/环境 | 8 卡映射、uv 装包、清华源、平台探测 |
| 2 | 模型路径简写 | `models/Qwen3.8` 不存在 → HFValidationError |
| 3 | torch.xpu 假象 | CUDA 兼容模式，一切写 cuda |
| 4 | torch.accelerator mock | 2fda97b 时代方案（回退后不再需要） |
| 5-8 | torch 2.5.1 兼容补丁链 | 2fda97b 时代补丁（回退后失效，vllm 0.15.1 原生支持） |
| 9 | **kunlun_ops 0.1.58 与 2fda97b 源码鸿沟** | **官方无新版 → 整体回退 4885de2 官方验证组合** |
| 10 | transformers 5.5.3 缺 `max_pixels` | 降级 5.2.0 |
| 11 | **采样器 exponential_ CPU fallback** | **decode 卡死根因 → kunlun_ops 设备采样 GPU 化**（附：KV 预算 OOM + random 参数陷阱） |

## 环境与平台（2026-08-15）

1. **容器 8 卡映射**：`--device=/dev/xpu0..7:/dev/xpu0..7` + `--device=/dev/xpuctrl`，
   漏映射卡数不对 → 一切分布式跑不起来
2. **模型路径**：简写 `models/Qwen3.8` 不存在 → vllm 走 snapshot_download 报
   HFValidationError 秒败。真实名 `models/Qwen3.8-27B-W8A8-INT8-Dynamic`
3. **torch.xpu.device_count()=0 是假象**：torch 原生模块 lru_cache 缓存 0，驱动正常。
   真验证：`torch_xmlir._XMLIRC._xpu_get_devices_number()` = 8
4. **昆仑芯 torch251 是 CUDA 兼容模式**：`torch/xpu/__init__.py` 是 Intel stub
   （`_is_compiled()` 恒 False）；真实后端 torch.cuda（`device="cuda:0"` 可建张量、
   103.1GB 显存、device_count=8）。vllm-kunlun 的 KunlunPlatform：
   `device_name="cuda"`、`dispatch_key="CUDA"`、`dist_backend="nccl"`、`is_cuda()=False`。
   **所有 device/张量/dist 代码写 cuda，勿写 xpu**

## torch 2.5.1 兼容补丁链（2026-08-16，仅 2fda97b 时代有效）

vllm 0.25.1（PyPI）要求 torch==2.11.0（`torch.accelerator.get_memory_info()` 2.6+ 才有）；
昆仑芯 xpytorch 只有 2.5.1 / 2.9 两代。0.25.1 + torch 2.5.1 组合需要 8 个补丁：

- **Patch2 torch.Size trace**（attention.py:527）
- **pass_manager 平台条件放宽**（is_cuda 专属区块不碰）
- **auto_functionalized 导出**（matcher_utils.py + `torch/_higher_order_ops/__init__.py`）
- **fusion passes 关闭**（enable_norm_fusion/enable_act_fusion 加昆仑芯判定
  device_name=="cuda" 且 is_cuda_alike()==False → False）

!!! info "回退后全部失效"
    vllm 0.15.1 **原生支持 torch 2.5.1**，patch_torch251 11/11 全适用，这 8 个补丁
    不再需要。此链仅作 2fda97b 时代历史。

## kunlun_ops 0.1.58 与 2fda97b 源码系统性不匹配（回退根因）

### 症状

vllm 0.25.1-dev（2fda97b）源码调用 kunlun_ops 0.1.58 时：

- `causal_conv1d` 关键字参数不匹配（KW_MISMATCH ×3）
- **11 个缺失算子**
- 接口鸿沟系统性 → 修不动

### 排查过程（结论）

kunlun_ops 0.1.58（2026-02-27）是**官方公开最新版**：

- 文档 stable/latest/main 三处全查 → 无新版
- docker hub 5 tags 全查 → 无新版
- base_v0.0.2 镜像用户确认 → 无新版

0.1.58 只匹配 **0.15.1.dev0 时代源码**（4885de2）。

### 决策

**整体回退官方验证组合**（issue #387 同款）：

```text
vllm-kunlun: 2fda97b (0.25.1-dev) → 4885de2 (0.15.1.dev0)
vllm:        0.25.1 → 0.15.1
transformers: 5.5.3 → 5.2.0
torch / kunlun_ops / triton / cocopod: 不动
```

重放脚本：`rollback_4885de2_part1.sh` + `rollback_4885de2_part3.sh`
（part2 有路径 bug 已并入 part3；备份在 `/home/newdata/backup/`，4.8M tar.gz）。

### 回退后验证

- patch_torch251：**11 applied | 0 failed**（对 vllm 0.15.x 写）
- `_kunlun` .so 位置：包根 `_kunlun.cpython-310-x86_64-linux-gnu.so`（13.5MB，
  不是 0.25.1 的 `_C/` 子目录）
- import vllm ✅

## transformers 5.5.3 缺 max_pixels API

### 症状

03 首次运行 `Engine core initialization failed`：

```text
AttributeError: 'Qwen2VLImageProcessor' object has no attribute 'max_pixels'
```

### 根因

- 容器 transformers=5.5.3（0.25.1 时代装的）
- 5.5.3 重构后移除 `Qwen2VLImageProcessor.max_pixels`
- vllm 0.15.1 `qwen2_vl.py:918`（MultiModalBudget）读该属性 → AttributeError

### 修复

```bash
uv pip install transformers==5.2.0 --no-deps --force-reinstall   # fix_tf_5_2_0.sh
```

!!! note "误判教训"
    类属性检查 `hasattr(Qwen2VLImageProcessor, 'max_pixels')` 返回 False 但实例属性
    存在（`__init__` 从 config 设置）——不必纠结，直接降级验证。

## 采样器 exponential_ CPU fallback（decode 卡死根因）

### 症状

04 压测 256 并发 decode 阶段 GPU 占用归零、吞吐卡死在 32 tok/s、进度条不动；
带 seed 的请求解码直接 AttributeError 崩溃。

### 根因

- `TopKTopPSampler` 无 k/p 过滤（top_k=0/top_p=1.0 归一化 → k/p=None）时走
  `forward_native` → `random_sample` 的 `q.exponential_()`
- 昆仑芯 CUDA 兼容层**没有 exponential_ 的设备 kernel** → 静默 CPU fallback
- 实测：256 batch × 152K vocab = **4.89s/step，GPU 0%**（probe_sampler.py 证据）

### 修复（commit b311b51）

`vllm_kunlun/v1/sample/ops/topk_topp_sampler.py`（容器 site-packages + 仓库根双份）：

1. **`forward_kunlun` 无 generators 一律走 `flashinfer_sample` 设备采样**，
   不再落 forward_native；k/p=None 时内部补 top_k=vocab / top_p=1.0
   （数学上等效无过滤纯随机采样）
2. **generators 分支**：`xspeedgate_ops.inplace_exponential` 在当前官方组合
   （xspeedgate_ops-0.0.0+torch25）中不存在（AttributeError）→ 统一改用
   `uniform_ + (-log)` 统计等价指数噪声（-log(U) ~ Exp(1)，FAST_RANDOM_SAMPLE
   同款算法；uniform_ 有设备 kernel，实测 0.01s）

### 效果

short 512/512 @256 并发 output 吞吐 32 → **3201.71 tok/s（~72 倍）**，
稳态峰值 4851 tok/s，RUN04_EXIT=0。

### 同轮附加坑（04 压测）

- **KV 预算 OOM**：`--gpu-memory-utilization 0.9` 把 KV 预分配顶满 86.24GiB/卡，
  GDN SSM state 4.69GiB @256 并发一分配即爆，8 卡全 OOM → **0.75**（KV ~72GiB，
  留 ~24GiB 给 SSM state + activation）
- **random 参数陷阱**：vllm 0.15.1 `--random-input-len/--random-output-len` 有
  默认值（1024/128）且优先于定长参数 → 04 脚本直传 random 版本
- **benchmark 入口**：v0.15.1 的 benchmark_throughput.py 是废弃占位（exit 1），
  入口 = `vllm bench throughput` CLI

## vllm 0.15.1 结构变化注意事项

回退后注意 vllm 0.15.1 与 0.25.1 的结构差异：

- config 是包（`vllm/config/cache.py` 等）
- `LLM.__init__` 是薄封装（显式参数少，**kwargs 透传 EngineArgs）——
  03 的 `enable_chunked_prefill` 等参数经透传可用（已实测验证）

## 操作教训

- **PowerShell 引号**：`\"` 被吞、单引号内嵌双引号传给远程 bash 被剥离、`|` 管道被
  PowerShell 解析 → **始终写 .sh 文件上传执行**
- **grep/ls 必须带 docker exec**：漏掉会在宿主机跑，报「没有那个文件或目录」虚惊
- **CRLF 陷阱**：本机 `git reset --hard` 在 .gitattributes 生效前跑过，工作树被
  autocrlf 转 CRLF（git 内仍 LF）→ 上传容器源码走 `git archive`，md5 比对去 `\r`
