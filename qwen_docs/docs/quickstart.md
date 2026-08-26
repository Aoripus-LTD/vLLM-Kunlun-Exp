# 快速开始

以下流程均已在生产服务器上验证通过（docker 容器 `qwen38-p800` 内）。
完整安装细节见 [安装](installation.md)，参数说明见 [启动参数](launch.md)。

## 1. 创建容器（8 卡全映射）

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

!!! note "操作要求"
    - 宿主机不执行任何 vllm 相关命令（仅允许 docker create/start/exec）
    - 容器创建后所有操作在容器内进行：`docker exec -it qwen38-p800 bash`
    - **8 张卡必须全部映射进容器**
    - 容器内 python 位于 `/opt/vllm_kunlun/bin/python`（venv，无 pip），包安装统一使用
      `uv pip install --python /opt/vllm_kunlun/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple`

## 2. 安装环境（0.1.122 栈）

```bash
# 驱动插件/算子栈（kunlun_ops 0.1.122 + triton + xspeedgate_ops + cocopod）
# cocopod 安装需 UV_SKIP_WHEEL_FILENAME_CHECK=1（wheel 文件名与内部版本号不一致）

# torch_xmlir：从 20260428/torch25 的 xpytorch .run 解压出 xmlir whl 后安装
uv pip install xmlir-1.0.0.1-cp310-cp310-linux_x86_64.whl --no-deps --force-reinstall

# vLLM（PyPI 直装）
uv pip install vllm==0.15.1 --force-reinstall --no-deps

# vllm-kunlun（在仓库根目录编译）
cd /home/newdata/vLLM-Kunlun-0.25.1-dev
python setup.py build && python setup.py install

# transformers（必须为 5.2.0）
uv pip install transformers==5.2.0 --no-deps --force-reinstall

# 补丁（vllm-kunlun 自带，针对 vllm 0.15.x：11 applied / 0 failed）
python vllm_kunlun/patches/patch_torch251.py
cp vllm_kunlun/patches/eval_frame.py /opt/vllm_kunlun/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py
cp vllm_kunlun/quantization/__init__.py /opt/vllm_kunlun/lib/python3.10/site-packages/vllm/model_executor/layers/quantization/__init__.py

# 启动前必加（xmlir 1.0.0.1 的 torch.compile 兼容开关）
export XMLIR_DYNAMO_WORKAROUND=1
```

!!! warning "editable 安装形态说明"
    vllm-kunlun 采用 editable 安装：`_editable_impl_vllm_kunlun.pth` 将仓库根加入
    sys.path，但实际加载路径为 site-packages/vllm_kunlun（可通过
    `python -c "import vllm_kunlun; print(vllm_kunlun.__file__)"` 验证）。
    源码同步需双份（site-packages 与仓库根）；编译必须在仓库根执行
    （构建文件位于仓库根，扩展源码在 `vllm_kunlun/csrc/`）。
    `_kunlun` 扩展位于包根 `_kunlun.cpython-310-x86_64-linux-gnu.so`（13.5MB）。

## 3. 环境变量（启动前 source） {#env-vars}

```bash
source /home/newdata/vLLM-Kunlun-0.25.1-dev/setup_env.sh
```

```text
XPU_VISIBLE_DEVICES=0-7
XFT_USE_FAST_SWIGLU=1
XMLIR_CUDNN_ENABLED=1
XPU_USE_DEFAULT_CTX=1
XMLIR_FORCE_USE_XPU_GRAPH=1
VLLM_HOST_IP=$(hostname -i)
VLLM_USE_V1=1
USE_ORI_ROPE=1        # Qwen3 融合大算子开关
```

## 4. 加载验证（03）

```python
# qwen38-deploy/03_vllm_load.py 核心参数
LLM(model="/home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic",
    tensor_parallel_size=8,          # TP=8 固定
    dtype='float16',                 # 与 config.dtype 一致
    quantization='kl3-compressed-xline',
    max_model_len=262144,            # 原生，无 YaRN
    mamba_ssm_cache_dtype='float16', # PR 408 必需
    enable_chunked_prefill=True,
    gpu_memory_utilization=0.9)
```

实测结果（2026-08-16）：

```text
[OK] 模型加载成功  (TP=8, kl3-compressed-xline, float16, max_len=262144)
[短生成]  你好，我是通义千问，由阿里巴巴通义实验室研发的大语言模型。
[长上下文] 模型回答: 200 TOPS        # 8000 token 前的事实，成功召回
[PASS] 召回结果: 包含正确数值 200
```

!!! warning "模型路径"
    真实目录名为 `models/Qwen3.8-27B-W8A8-INT8-Dynamic`（约 30GB，含同名 .zip）。
    简写 `models/Qwen3.8` 不存在，vllm 会触发 snapshot_download 并报 HFValidationError。

## 5. 吞吐压测（04，已验证通过）

```bash
# short 组：512/512，256 prompts（对齐官方基准口径）
python 04_throughput_bench.py --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic --group short
# long 组：32768/256，16 prompts
python 04_throughput_bench.py --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic --group long --num-prompts 16
```
