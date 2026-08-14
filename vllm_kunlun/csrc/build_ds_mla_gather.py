#!/usr/bin/env python3
"""Build + load the native ds_mla_gather custom op (torch_xmlir libxmlir_custom_ops).

Run inside the kunlun-bench container with the Kunlun venv:

    LD_LIBRARY_PATH=/opt/vllm_kunlun/lib/python3.10/site-packages/torch_xmlir:/opt/vllm_kunlun/xcudart/lib:$LD_LIBRARY_PATH \
        DISABLE_XPYTORCH=1 /opt/vllm_kunlun/bin/python build_ds_mla_gather.py

It JIT-compiles ds_mla_gather.cpp with g++ (torch's CppExtension), links
against torch + torch_xmlir's libxmlir_custom_ops.so, loads the result and
runs a tiny correctness self-check.
"""
import os
import sys

import torch
from torch.utils.cpp_extension import load

TORCH_XMLIR = "/opt/vllm_kunlun/lib/python3.10/site-packages/torch_xmlir"
XCUDART = "/opt/vllm_kunlun/xcudart"

_here = os.path.dirname(os.path.abspath(__file__))

lib_path = load(
    name="ds_mla_gather",
    sources=[os.path.join(_here, "ds_mla_gather.cpp")],
    is_python_module=False,
    extra_include_paths=[
        os.path.join(TORCH_XMLIR, "include"),
        os.path.join(XCUDART, "include"),
        "/opt/vllm_kunlun/lib/python3.10/site-packages/triton/backends/nvidia/include",
    ],
    extra_ldflags=[
        f"-L{TORCH_XMLIR}",
        f"-L{XCUDART}/lib",
        f"-Wl,-rpath,{TORCH_XMLIR}",
        f"-Wl,-rpath,{XCUDART}/lib",
        "-lxmlir_custom_ops",
        "-lxpuapi",
        "-lxdnn_pytorch",
        "-lcudart",
        "-ltorch_cuda",
        "-lc10_cuda",
    ],
    verbose=False,
)
print("built:", lib_path)
torch.ops.load_library(lib_path)
print("ds_mla_gather loaded + registered OK (torch.ops.ds_mla_gather.*)")
print("LIB_PATH=" + lib_path)
