# 性能与压测

## 吞吐目标

- 单流 **≥ 30 tok/s**
- 几千并发下 **5000+ tok/s**（官方 R1-671B INT8 口径）

## XCCL 实测（2026-08-16，01_env_check）

- backend=`nccl` → 昆仑芯 ProcessGroupXCCL（底层 libbkcl.so），8 设备
- allreduce 2.6MB 单轮 **0.13ms → 19.3 GB/s**
- 每 token 通信 = 64 层 × 0.13ms ≈ **8.32ms**
- TP=8 单流理论 ~9.7ms/token（权重读 1.4ms + 通信 8.32ms）

## 官方千帆表（TP=8 单实例口径，@256 并发）

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
| 完成时间 | ~28s，256/256，无卡死 |
| KV 占用峰值 | 9.7% |

**long 组**（input 32768 / output 256 × 16 并发）：

| 指标 | 数值 |
|---|---|
| 吞吐 | 0.33 req/s · 84.02 output tok/s（total 10839.18） |
| prefill 峰值 | 9829.5 input tok/s（524288 tokens 主导耗时） |
| 完成时间 | ~45s（~40s prefill + ~4s 生成），16/16 |
| KV 占用峰值 | 12.3% |

**解读**：

- 生成稳态峰值 **4851 output tok/s**，已接近官方 5000+ 口径（R1-671B INT8
  256 并发 2437 tok/s 的两倍，同代 32B 级模型可比）
- 修复前（exponential_ CPU fallback）同场景卡死 32 tok/s，GPU 化后提升
  **~72 倍**，详见 [故障排查](troubleshooting.md)
- long 组 prefill 主导：32K 上下文下 chunked prefill 抢占算力，生成阶段
  仅 ~4s；业务上长上下文场景关注 TTFT 而非生成吞吐

## 带宽预算（27B W8A8，权重 ~31GB = I8 24G + F16 7G）

- **单卡单流 decode**：31GB / 2.4TB/s ≈ 12.9ms/token → ~77 tok/s 带宽上限
- **单卡多并发**：batch 放大下权重读取共享，受算力 345T ÷ 54GFLOP/token ≈ 6400 tok/s 二次约束
- **TP=8 单流 decode**：每卡读 3.4GB ≈ 1.4ms + allreduce 通信（2.6MB/token ÷ 64 层 ≈ 40KB/层/卡）
- **TP=8 vs 单卡分界**：TP=8 单流 ≈ 1.4ms + 2.6MB×64÷BW ≤ 12.9ms ⟺ **BW ≥ ~15 GB/s**
  → 实测 19.3 GB/s > 15 GB/s，TP=8 峰值更高（8 卡并行读权重）
- **256K 长上下文（单卡）**：权重 31G + KV 16.8G ≈ 48G/token → 单流 ~51 tok/s；
  单卡 96G（util 0.9 ≈ 86G）可容 3 个 256K 并发 → 每卡 ~150 tok/s

## 部署形态决策（2026-08-16 用户最终决策）

**只跑 TP=8 单实例**（取消双形态对比、取消单卡多实例）。依据：

1. **官方千帆表 = TP=8 单实例口径**：R1-671B INT8 ≈671GB 必须 TP=8（单卡装不下），
   整表统一测法 → 官方 Dense 高分是 TP=8 在 P800 的实证
2. **Dense allreduce 量小**：每层仅几十 KB，XCCL 19.3GB/s 扛得住（官方实证）
3. **用户实测「单卡 > 多卡 MoE」**：是 MoE all-to-all 通信烂（token 搬运量巨大），
   与 Dense 不矛盾

判定表（历史背景，已不执行）：BW≥100→TP=8；15-100→双跑；<15→单卡。

## 压测方法（04_throughput_bench.py，仅 TP=8）

```bash
# short 组：input 512 / output 512，256 prompts（对齐官方 R1 5000 tok/s 基准口径）
python 04_throughput_bench.py --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic --group short

# long 组：input 32768 / output 256，16 prompts（业务长上下文吞吐观测）
python 04_throughput_bench.py --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic --group long --num-prompts 16
```

复用 vllm 官方 benchmark_throughput（v0.15.1 入口为 `vllm bench throughput` CLI，
benchmark_throughput.py 是废弃占位），输出 requests/s 与 tokens/s。

**参数陷阱（vllm 0.15.1）**：`--random-input-len/--random-output-len` 有默认值
（1024/128）且 random 版本优先于定长 `--input-len/--output-len`——04 脚本内部
直接传 random 版本参数（`--random-range-ratio` 默认 0.0 = 固定长度），避免
被默认值覆盖导致实际压测长度与预期不符。

**KV 预算**：`--gpu-memory-utilization 0.75`（默认 0.9 会把 KV 预分配顶满
86.24GiB/卡，GDN SSM state @256 并发一分配即 OOM，8 卡全爆实证）。
