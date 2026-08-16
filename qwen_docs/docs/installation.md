# 安装

逐步安装详解。**先决条件**：容器已创建（见 [快速开始](quickstart.md)），8 卡全映射，
数据路径 `/home/newdata`（14T LVM，docker data-root = `/home/docker`，无需改 daemon.json）。

## 版本对照（官方验证组合，issue #387 同款）

| 组件 | 版本 | 安装方式 |
|---|---|---|
| torch | 2.5.1 | xpytorch `.run` 直装（`bash xpytorch-xxx.run`） |
| kunlun_ops | 0.1.58 | whl：`uv pip install kunlun_ops-0.1.58+ee39020a` |
| triton | 3.0.0+b2cde523 | whl |
| xspeedgate_ops | 0.0.0+torch25 | whl |
| cocopod | 0.0.0+torch25 | whl（**需 `UV_SKIP_WHEEL_FILENAME_CHECK=1`**） |
| vllm | 0.15.1 | PyPI：`uv pip install vllm==0.15.1 --force-reinstall --no-deps` |
| vllm-kunlun | 0.15.1.dev0（4885de2） | 仓库根 `python setup.py build && python setup.py install` |
| transformers | 5.2.0 | PyPI：`uv pip install transformers==5.2.0 --no-deps --force-reinstall` |

!!! danger "版本对应铁律"
    **kunlun_ops 0.1.58 只匹配 0.15.1.dev0 时代源码**。若装 vllm-kunlun 0.25.1-dev
    （2fda97b），会遇 causal_conv1d 关键字参数、11 个缺失算子、3 个 KW_MISMATCH——
    接口鸿沟系统性，修不动，只能整体回退（见 [故障排查](troubleshooting.md)）。

## Step 1：装包

```bash
PY=/opt/vllm_kunlun/bin/python
UV="/root/.local/bin/uv pip install --python $PY --index-url https://pypi.tuna.tsinghua.edu.cn/simple"

# 算子栈
$UV kunlun_ops-0.1.58+ee39020a.whl
$UV triton-3.0.0+b2cde523.whl
$UV xspeedgate_ops-0.0.0+torch25.whl
UV_SKIP_WHEEL_FILENAME_CHECK=1 $UV cocopod-0.0.0+torch25.whl

# vLLM + transformers
$UV vllm==0.15.1 --force-reinstall --no-deps
$UV transformers==5.2.0 --no-deps --force-reinstall
```

## Step 2：编译 vllm-kunlun

```bash
cd /home/newdata/vLLM-Kunlun-0.25.1-dev     # 仓库根（构建文件在这里）
python setup.py build
python setup.py install
```

- `_kunlun` 扩展产物：包根 `_kunlun.cpython-310-x86_64-linux-gnu.so`（13.5MB）
- **editable 陷阱**：import 实际加载 site-packages/vllm_kunlun（`.pth` 被遮蔽），
  源码同步要双份（site-packages + 仓库根）

## Step 3：补丁

```bash
cd /home/newdata/vLLM-Kunlun-0.25.1-dev

# torch 2.5.1 兼容补丁（vllm-kunlun 自带，对 vllm 0.15.x 写 → 11 applied / 0 failed）
python vllm_kunlun/patches/patch_torch251.py

# eval_frame + quantization 替换（vllm-kunlun 的昆仑芯实现）
SP=/opt/vllm_kunlun/lib/python3.10/site-packages
cp vllm_kunlun/patches/eval_frame.py $SP/torch/_dynamo/eval_frame.py
cp vllm_kunlun/quantization/__init__.py $SP/vllm/model_executor/layers/quantization/__init__.py
```

## Step 4：验证安装

```bash
PY=/opt/vllm_kunlun/bin/python

# torch + 加速器
$PY -c "import torch; print(torch.__version__, hasattr(torch, 'accelerator'))"

# 设备数（真验证，torch.xpu.device_count()=0 是假象）
$PY -c "import torch_xmlir; print(torch_xmlir._XMLIRC._xpu_get_devices_number())"   # 期望 8

# vllm-kunlun 加载位置（应为 site-packages）
$PY -c "import vllm_kunlun; print(vllm_kunlun.__file__)"

# _kunlun 扩展
$PY -c "from vllm_kunlun import _kunlun; print('KUNLUN_SO_OK')"

# XCCL 通信（allreduce 微基准，期望 19.3 GB/s）
$PY xccl_bench.py
```

## Step 5：环境变量

```bash
source /home/newdata/vLLM-Kunlun-0.25.1-dev/setup_env.sh
```

变量清单见 [快速开始 §3](quickstart.md#env-vars)。
