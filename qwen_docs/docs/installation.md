# 安装

逐步安装详解。**先决条件**：容器已创建（见 [快速开始](quickstart.md)），8 卡全映射，
数据路径 `/home/newdata`（14T LVM，docker data-root 为 `/home/docker`，无需修改 daemon.json）。

## 版本对照（2026-08-26 更新）

| 组件 | 版本 | 安装方式 |
|---|---|---|
| torch | 2.5.1 | xpytorch `.run` 直装（`bash xpytorch-xxx.run`） |
| torch_xmlir | xmlir 1.0.0.1（**20260428 版**） | 解压 20260428/torch25 xpytorch `.run` 后 `uv pip install --no-deps --force-reinstall xmlir-1.0.0.1-*.whl` |
| kunlun_ops | 0.1.122 | whl：`uv pip install kunlun_ops-0.1.122+b4984657` |
| triton | 3.0.0+b2cde523 | whl |
| xspeedgate_ops | 1.5.0+torch25 | whl |
| cocopod | 0.0.0+torch25 | whl（需 `UV_SKIP_WHEEL_FILENAME_CHECK=1`） |
| vllm | 0.15.1 | PyPI：`uv pip install vllm==0.15.1 --force-reinstall --no-deps` |
| vllm-kunlun | 0.15.1.dev0（4885de2 + MTP patches） | 仓库根 `python setup.py build && python setup.py install` |
| transformers | 5.2.0 | PyPI：`uv pip install transformers==5.2.0 --no-deps --force-reinstall` |

!!! danger "版本兼容性要求"
    - kunlun_ops 0.1.58 仅匹配 0.15.1.dev0 时代源码（旧 dense 验证组合）。
    - **kunlun_ops 0.1.122 必须搭配 xmlir 1.0.0.1 的 20260428 版**：更早的
      0409 版缺少 31 参 `xfa::gated_delta_net` 符号，0.1.122 import 报 undefined
      symbol；同时 0.1.122 的 Python 扩展 `xpu_kunlun_ops` / `xpu_flash_ops`
      位于 whl 根目录（site-packages 顶层），安装时不要遗漏。
    - xmlir 升级后启动必须 `export XMLIR_DYNAMO_WORKAROUND=1`，否则
      torch.compile 在 `make_tensors_stateful` 上 Unsupported（见
      [启动参数](launch.md)）。

## Step 1：装包

```bash
PY=/opt/vllm_kunlun/bin/python
UV="/root/.local/bin/uv pip install --python $PY --index-url https://pypi.tuna.tsinghua.edu.cn/simple"

# 算子栈
$UV kunlun_ops-0.1.122+b4984657.whl
$UV triton-3.0.0+b2cde523.whl
$UV xspeedgate_ops-1.5.0+torch25.whl
UV_SKIP_WHEEL_FILENAME_CHECK=1 $UV cocopod-0.0.0+torch25.whl

# torch_xmlir（从 20260428/torch25 xpytorch .run 解压出的 xmlir whl）
$UV xmlir-1.0.0.1-cp310-cp310-linux_x86_64.whl --no-deps --force-reinstall

# vLLM + transformers
$UV vllm==0.15.1 --force-reinstall --no-deps
$UV transformers==5.2.0 --no-deps --force-reinstall
```

## Step 2：编译 vllm-kunlun

```bash
cd /home/newdata/vLLM-Kunlun-0.25.1-dev     # 仓库根（构建文件所在目录）
python setup.py build
python setup.py install
```

- `_kunlun` 扩展产物：包根 `_kunlun.cpython-310-x86_64-linux-gnu.so`（13.5MB）
- **editable 安装形态说明**：import 实际加载 site-packages/vllm_kunlun（`.pth` 被遮蔽），
  源码同步需双份（site-packages 与仓库根）

## Step 3：补丁

```bash
cd /home/newdata/vLLM-Kunlun-0.25.1-dev

# torch 2.5.1 兼容补丁（vllm-kunlun 自带，针对 vllm 0.15.x：11 applied / 0 failed）
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

# 设备数（torch.xpu.device_count() 返回 0 不具备参考意义，以此接口为准）
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
