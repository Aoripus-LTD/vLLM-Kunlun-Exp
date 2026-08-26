# 性能与压测

## 吞吐目标

- 单流 **≥ 30 tok/s**
- 数千并发下 **5000+ tok/s**（官方 R1-671B INT8 口径）

## XCCL 实测（2026-08-16，01_env_check）

- backend=`nccl`，对应昆仑芯 ProcessGroupXCCL（底层 libbkcl.so），8 设备
- allreduce 2.6MB 单轮 **0.13ms，约 19.3 GB/s**
- 每 token 通信 = 64 层 × 0.13ms ≈ **8.32ms**
- TP=8 单流理论时延约 9.7ms/token（权重读取 1.4ms + 通信 8.32ms）

## 官方性能基准（TP=8 单实例口径，256 并发）

| 模型 | tok/s |
|---|---|
| R1-671B INT8 | 2437 |
| Distill-70B | 4185 |
| Distill-32B | 10328 |
| Distill-14B | 18296 |

## 实测结果（2026-08-16 15:35 权威参数轮次，官方验证组合）

官方组合 = torch 2.5.1 + kunlun_ops 0.1.58 + vllm 0.15.1 + vllm-kunlun 4885de2
+ transformers 5.2.0，TP=8，`--gpu-memory-utilization 0.75`。

**short 组**（input 512 / output 512 × 256 并发，对齐官方基准口径）：

| 指标 | 数值 |
|---|---|
| 吞吐 | 6.25 req/s · **3201.71 output tok/s**（total 6403.43） |
| 稳态峰值 | 4851.1 output tok/s（满 batch 纯生成阶段） |
| 完成时间 | 约 28s，256/256 全部完成 |
| KV 占用峰值 | 9.7% |

**long 组**（input 32768 / output 256 × 16 并发）：

| 指标 | 数值 |
|---|---|
| 吞吐 | 0.33 req/s · 84.02 output tok/s（total 10839.18） |
| prefill 峰值 | 9829.5 input tok/s（524288 tokens 主导耗时） |
| 完成时间 | 约 45s（约 40s prefill + 约 4s 生成），16/16 全部完成 |
| KV 占用峰值 | 12.3% |

**结果分析**：

- 生成稳态峰值 **4851 output tok/s**，已接近官方 5000+ 口径（为 R1-671B INT8
  256 并发 2437 tok/s 的两倍，与同代 32B 级模型可比）
- 修复前（exponential_ CPU fallback）同场景下吞吐停滞于 32 tok/s，改为设备端
  采样后提升约 **72 倍**，详见 [故障排查](troubleshooting.md)
- long 组以 prefill 为主导：32K 上下文下 chunked prefill 占用主要算力，生成阶段
  仅约 4s；长上下文场景应关注 TTFT 而非生成吞吐

## 实测结果（2026-08-26，kunlun_ops 0.1.122 栈 + MTP）

算子栈升级（kunlun_ops 0.1.122 + xmlir 1.0.0.1 20260428 版）后，API 层 overall
吞吐（prompt 172 + output 256）：

| 形态 | 单流 overall | 8 并发 | 32 并发 | 256 并发 |
|---|---|---|---|---|
| Dense（0.1.58 栈） | 57 | 220 | 665 | 1542 |
| Dense（0.1.122 栈） | 52 | — | — | **1552** |
| **MTP（0.1.122 栈）** | **55-62** | 246 | 709 | 1482 |

- MTP 配置：`--speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}'`，
  Mean acceptance length **1.83**
- 单流 MTP 比 Dense 高约 9%（投机解码摊薄 GDN 串行成本）；高并发下 draft 计算
  与验证争抢算力，比 Dense 低约 4.5%——按负载特征选择形态
- 升级要点：xmlir 必须用 **20260428 版**（0409 版缺 31 参 `xfa::gated_delta_net`），
  且启动前 `export XMLIR_DYNAMO_WORKAROUND=1`（详见 [安装](installation.md)）
- MTP 与 mamba prefix-caching 不可共存（vllm 0.15.1 限制）

## 带宽预算（27B W8A8，权重约 31GB = I8 24G + F16 7G）

- **单卡单流 decode**：31GB / 2.4TB/s ≈ 12.9ms/token，约 77 tok/s 带宽上限
- **单卡多并发**：batch 放大下权重读取共享，受算力约束 345T ÷ 54GFLOP/token ≈ 6400 tok/s
- **TP=8 单流 decode**：每卡读取 3.4GB ≈ 1.4ms + allreduce 通信（2.6MB/token ÷ 64 层 ≈ 40KB/层/卡）
- **TP=8 与单卡分界**：TP=8 单流 ≈ 1.4ms + 2.6MB×64÷BW ≤ 12.9ms，等价于 **BW ≥ 约 15 GB/s**；
  实测 19.3 GB/s 高于该临界值，TP=8 峰值更高（8 卡并行读取权重）
- **256K 长上下文（单卡）**：权重 31G + KV 16.8G ≈ 48G/token，单流约 51 tok/s；
  单卡 96G（util 0.9 约 86G）可容纳 3 个 256K 并发，每卡约 150 tok/s

## 部署形态决策（2026-08-16 最终决策）

**仅采用 TP=8 单实例**（不再考虑双形态对比与单卡多实例）。依据：

1. **官方性能基准为 TP=8 单实例口径**：R1-671B INT8 约 671GB，单卡无法容纳，必须
   TP=8；整表采用统一测试方法，官方 Dense 高分即为 TP=8 在 P800 上的实证
2. **Dense 模型 allreduce 通信量小**：每层仅几十 KB，实测 19.3GB/s 可满足
3. **「单卡优于多卡 MoE」的实测结论**源于 MoE all-to-all 通信开销（token 搬运量
   巨大），与 Dense 模型不矛盾

决策阈值表（历史背景，已不再执行）：BW≥100 → TP=8；15-100 → 双形态对比；<15 → 单卡。

## 压测方法（04_throughput_bench.py，仅 TP=8）

```bash
# short 组：input 512 / output 512，256 prompts（对齐官方 R1 5000 tok/s 基准口径）
python 04_throughput_bench.py --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic --group short

# long 组：input 32768 / output 256，16 prompts（业务长上下文吞吐观测）
python 04_throughput_bench.py --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic --group long --num-prompts 16
```

复用 vllm 官方 benchmark_throughput（v0.15.1 入口为 `vllm bench throughput` CLI，
benchmark_throughput.py 为废弃占位），输出 requests/s 与 tokens/s。

**参数注意事项（vllm 0.15.1）**：`--random-input-len/--random-output-len` 存在默认值
（1024/128）且 random 版本优先于定长 `--input-len/--output-len`——04 脚本内部
直接传 random 版本参数（`--random-range-ratio` 默认 0.0，即固定长度），避免
被默认值覆盖导致实际压测长度与预期不符。

**KV 预算**：`--gpu-memory-utilization 0.75`（默认 0.9 会将 KV 预分配占满
86.24GiB/卡，GDN SSM state 在 256 并发下分配即触发 OOM，8 卡全部复现该问题）。
