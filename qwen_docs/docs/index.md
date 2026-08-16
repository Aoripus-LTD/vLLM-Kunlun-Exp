# Qwen3.8-27B on Kunlun P800 部署手册

单机 **8×96G 昆仑芯 P800-OAM** 通过 vllm-kunlun 部署 **Qwen3.8-27B-INT8-W8A8-Dynamic**
（Aoripus-KL3-XLine v2.2-Dev-Nightly 量化导出），原生 **262144（256K）上下文**，**TP=8 单实例**。

## 当前状态

=== "2026-08-16：03 加载验证通过 🎉"

    ```text
    [OK] 模型加载成功  (TP=8, kl3-compressed-xline, float16, max_len=262144)
    [短生成]  你好，我是通义千问，由阿里巴巴通义实验室研发的大语言模型。
    [长上下文] 模型回答: 200 TOPS        # 8000 token 前的事实，成功召回
    [PASS] 召回结果: 包含正确数值 200
    ```

=== "已定稿决策"

    - **部署形态**：TP=8 单实例（取消双形态对比、取消单卡多实例）
    - **上下文**：原生 262144，不做 YaRN 1M 扩展（业务需求 ≤256K）
    - **环境组合**：torch 2.5.1 + kunlun_ops 0.1.58 + vllm 0.15.1 + vllm-kunlun 4885de2 + transformers 5.2.0（官方验证组合）
    - **压测**：04 已通过（short 3201.71 output tok/s / long 84.02 output tok/s）

## 文档导航

| 章节 | 内容 |
|---|---|
| [项目概览](overview.md) | 项目定位、硬件/软件规格、版本对应铁律 |
| [快速开始](quickstart.md) | 容器创建 → 环境安装 → 启动验证 全流程 |
| [安装](installation.md) | 逐步安装详解（官方验证组合） |
| [模型与适配](model.md) | Qwen3.8 架构、W8A8 量化格式、config 权威字段 |
| [启动参数](launch.md) | 权威参数说明（float16 / 262144 / PR 408） |
| [性能与压测](performance.md) | XCCL 实测、带宽预算、官方口径、压测方法 |
| [故障排查](troubleshooting.md) | 部署全程问题记录与修复链 |
| [部署资产](assets.md) | qwen38-deploy/ 脚本清单与用法 |
