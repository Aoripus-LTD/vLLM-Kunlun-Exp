#!/bin/bash
# 01_env_check_v3.sh — 环境自检 + XCCL(allreduce) 实测 v3
# v3 修正（v2 的 xpu:0 失败根因）：KunlunPlatform 是 CUDA 模式运行
#   (kunlun.py: device_name="cuda", dist_backend="nccl")，
#   torch.xpu 是上游 Intel stub（未编译），真实后端是 torch.cuda。
# 宿主机执行，docker exec -i 进容器，输出 env_report.txt
OUT="env_report.txt"
{
echo "============== 容器内 Python 环境 =============="
docker exec qwen38-p800 /opt/vllm_kunlun/bin/python -c "import torch; print('torch:', torch.__version__)" 2>&1 | grep -v -E "XCCL|SYMBOL|UserWarning|pkg_resources"
docker exec qwen38-p800 /opt/vllm_kunlun/bin/pip list 2>/dev/null | grep -i -E "vllm|kunlun|torch|triton|transformers|compressed|cocopod|xspeedgate"

echo "============== XCCL allreduce 实测（TP=8 决策关键） =============="
echo "[说明] 测 27B Dense TP=8 每 token allreduce 量(~2.6MB) 的实测带宽"
echo "[说明] 昆仑芯 torch251 为 CUDA 兼容模式: device=cuda, backend=nccl(底层BKCL)"
docker exec -i qwen38-p800 /opt/vllm_kunlun/bin/python - <<'PYEOF'
import os
import time

import torch
import torch.distributed as dist
import torch_xmlir._XMLIRC as xm

# 真设备数: torch.xpu.device_count() 是 Intel stub 恒 0, 必须用 _XMLIRC
ndev = xm._xpu_get_devices_number()
print("  XPU 真设备数 (_XMLIRC):", ndev)
print("  torch.cuda.device_count():", torch.cuda.device_count())

# 验证 cuda 后端（= 昆仑芯真实算力后端）可建张量
try:
    t0 = torch.zeros(1, device="cuda:0")
    print("  cuda:0 张量 OK:", t0.device)
    print("  cuda:0 显存:", round(torch.cuda.mem_get_info(0)[1] / 1e9, 1), "GB 总量")
except Exception as e:
    print("  [FAIL] cuda 张量创建失败:", e)
    raise SystemExit(1)


def worker(rank, world, port):
    dev = f"cuda:{rank}"
    backend = None
    for b in ("nccl", "bkcl", "xccl"):
        try:
            dist.init_process_group(backend=b,
                                    init_method=f"tcp://127.0.0.1:{port}",
                                    rank=rank, world_size=world)
            backend = b
            break
        except Exception as e:
            if rank == 0:
                print(f"  backend={b} init 失败: {str(e)[:200]}")
    if backend is None:
        if rank == 0:
            print("  [FAIL] 所有 backend 初始化失败，XCCL 实测需人工补测")
        return
    nbytes = 2_600_000 // 4 * 4  # 27B Dense TP=8 per-token allreduce 量
    t = torch.randn(nbytes // 4, device=dev)
    dist.barrier()
    for _ in range(5):
        dist.all_reduce(t)
    dist.barrier()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(t)
    dt = (time.perf_counter() - t0) / iters
    if rank == 0:
        bw = nbytes / dt / 1e9
        print(f"  [OK] backend={backend} 设备数={world}")
        print(f"  [XCCL] allreduce 2.6MB 单轮 {dt*1e3:.2f}ms → 实测带宽 {bw:.1f} GB/s")
        print(f"  [对比] 单卡 HBM3 = 2400 GB/s；allreduce 带宽占比 {bw/2400*100:.1f}%")
        print(f"  [判断] 27B TP=8 每 token 通信 2.6MB×64 层 → 每 token 通信耗时 "
              f"{2.6e6/dt/1e9*64:.2f}ms")
    dist.destroy_process_group()


if ndev < 2:
    print("  设备 <2，跳过 allreduce 实测（单卡部署无需跨卡）")
else:
    world = min(ndev, 8)
    port = 24000 + os.getpid() % 1000
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(world, port), nprocs=world, join=True)
PYEOF
echo "[完成] 报告已写入: $(pwd)/$OUT"
} 2>&1 | tee "$OUT"
