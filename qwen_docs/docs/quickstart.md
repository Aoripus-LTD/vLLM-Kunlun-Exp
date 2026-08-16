# 快速开始

以下流程均为**已在生产服务器实际跑通**的步骤（docker 容器 `qwen38-p800` 内）。
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

!!! note "执行铁律"
    - 宿主机**绝不执行任何 vllm 相关命令**（只允许 docker create/start/exec）
    - 创建容器后所有操作都在容器内：`docker exec -it qwen38-p800 bash`
    - **8 张卡必须全部映射进容器**
    - 容器内 python 是 `/opt/vllm_kunlun/bin/python`（venv，**无 pip**）
      → 装包一律 `uv pip install --python /opt/vllm_kunlun/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple`

## 2. 安装环境（官方验证组合）

```bash
# 驱动插件/算子栈（kunlun_ops 0.1.58 + triton + xspeedgate_ops + cocopod）
# cocopod 需 UV_SKIP_WHEEL_FILENAME_CHECK=1（文件名与内部版本号不符）

# vLLM（PyPI 直装）
uv pip install vllm==0.15.1 --force-reinstall --no-deps

# vllm-kunlun（仓库根编译，必须 cd 仓库根）
cd /home/newdata/vLLM-Kunlun-0.25.1-dev
python setup.py build && python setup.py install

# transformers（必须是 5.2.0）
uv pip install transformers==5.2.0 --no-deps --force-reinstall

# 补丁（vllm-kunlun 自带，对 vllm 0.15.x 写 → 11 applied / 0 failed）
python vllm_kunlun/patches/patch_torch251.py
cp vllm_kunlun/patches/eval_frame.py /opt/vllm_kunlun/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py
cp vllm_kunlun/quantization/__init__.py /opt/vllm_kunlun/lib/python3.10/site-packages/vllm/model_executor/layers/quantization/__init__.py
```

!!! warning "editable 安装形态陷阱"
    vllm-kunlun 是 editable 安装：`_editable_impl_vllm_kunlun.pth` 加 sys.path 但被
    site-packages 遮蔽 → **import 实际加载 site-packages/vllm_kunlun**（用
    `python -c "import vllm_kunlun; print(vllm_kunlun.__file__)"` 验证）。
    源码同步要双份：site-packages + 仓库根；编译必须 cd 仓库根
    （构建文件在仓库根不在包内，扩展源在 `vllm_kunlun/csrc/`）。
    `_kunlun` 扩展在包根 `_kunlun.cpython-310-x86_64-linux-gnu.so`（13.5MB）。

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
    dtype='float16',                 # = config.dtype
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
    真实目录名为 `models/Qwen3.8-27B-W8A8-INT8-Dynamic`（~30GB 含同名 .zip）。
    简写 `models/Qwen3.8` **不存在** → vllm 会走 snapshot_download 报 HFValidationError。

## 5. 吞吐压测（04，已实测通过）

```bash
# short 组：512/512，256 prompts（对齐官方口径）
python 04_throughput_bench.py --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic --group short
# long 组：32768/256，16 prompts
python 04_throughput_bench.py --model /home/newdata/models/Qwen3.8-27B-W8A8-INT8-Dynamic --group long --num-prompts 16
```
