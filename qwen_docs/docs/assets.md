# 部署资产

所有脚本位于仓库内 `qwen38-deploy/` 目录。

## 核心脚本

| 脚本 | 用途 | 运行位置 |
|---|---|---|
| `01_env_check_v3.sh` + `xccl_bench.py` | 环境自检 + XCCL allreduce 带宽基准 | 服务器（容器内） |
| `03_vllm_load.py` | 加载验证 + 短/长上下文召回 | 服务器（容器内） |
| `04_throughput_bench.py` | 吞吐压测（short/long 两组，仅 TP=8） | 服务器（容器内） |
| `run03.sh` / `run04.sh` + `watch_03_run.py` / `watch04.py` | 03/04 编排 + 日志轮询 | 服务器（容器内）/ 本机 |

## 修复/回退脚本（可重放）

| 脚本 | 用途 |
|---|---|
| `rollback_4885de2_part1.sh` | 回退 part1：备份 2fda97b → 解压 4885de2 → 替换 site-packages → 安装 vllm 0.15.1 |
| `rollback_4885de2_part2.sh` | 回退 part2：仓库根备份 + 覆盖（路径问题已并入 part3） |
| `rollback_4885de2_part3.sh` | 回退 part3：仓库根编译 _kunlun → 补丁（eval_frame/quantization/patch_torch251）→ import 验证 |
| `fix_tf_5_2_0.sh` | transformers 5.5.3 → 5.2.0（缺失 max_pixels API） |

## 验证/诊断脚本

| 脚本 | 用途 |
|---|---|
| `verify_rollback.sh` | 回退后验证（.so 位置 + setup_env.sh + import） |
| `probe_rollback.sh` | editable 安装形态探测 |
| `probe_tf.sh` | transformers 版本 + max_pixels 探测 |
| `check_llm_sig.sh` / `check_vllm_params.sh` | LLM 签名 / CacheConfig 字段检查 |
| `watch_03_run.py` | 03 运行轮询 watcher（paramiko 版） |

## 同步/管理脚本

| 脚本 | 用途 |
|---|---|
| `ssh_run.py` | paramiko SSH 执行/上传（连接参数全部走环境变量） |
| `hash_verify.sh` | 容器与本机源码 md5 清单比对（LF 级） |
| `sync_vllm_kunlun.sh` | 源码同步（备份 → 覆盖 → 重编译 _kunlun → 重拷补丁 → 验证） |
| `fetch_docs.py` | readthedocs 全站爬取器（本地文档索引） |

## SSH 工具用法（ssh_run.py）

```powershell
# PowerShell（本机）
$env:QWEN38_HOST='服务器地址'
$env:QWEN38_PORT='端口'            # 可选，默认 22
$env:QWEN38_USER='root'            # 可选，默认 root
$env:QWEN38_SSH_PASS='密码'
python ssh_run.py "docker exec -it qwen38-p800 bash -lc '命令'"
```

!!! warning "ssh_run.py 注意事项"
    - 连接参数与凭据**绝不入库**，全部从环境变量读取（QWEN38_HOST / QWEN38_PORT / QWEN38_USER / QWEN38_SSH_PASS）
    - stdout 含 emoji 时需加 `PYTHONIOENCODING=utf-8` 前缀（Windows GBK 下会
      UnicodeEncodeError）
    - **多余位置参数会追加进远程命令**——不要传入多余参数
    - 参数不要使用双引号（PowerShell 会吞引号）——复杂命令写 .sh 上传执行

## 03 加载验证脚本参数（权威）

```bash
python 03_vllm_load.py --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic
```

脚本内置定稿参数：TP=8 固定、`--max-model-len 262144`、无 YaRN、`--dtype float16`、
`--quantization kl3-compressed-xline`、`--mamba-ssm-cache-dtype float16`。
