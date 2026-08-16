#!/usr/bin/env python3
# xccl_bench.py — 昆仑芯 P800 XCCL(allreduce) 微基准
# 测 27B Dense TP=8 每 token allreduce 量(~2.6MB) 实测带宽 → TP=8 决策
# 昆仑芯 torch251 为 CUDA 兼容模式: device=cuda, backend=nccl(底层BKCL)
# 注意: 必须写成真实文件执行 (mp.spawn spawn 模式需 re-import 主文件, heredoc 不行)
import os
import time

import torch
import torch.distributed as dist
import torch_xmlir._XMLIRC as xm


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


if __name__ == "__main__":
    # 真设备数: torch.xpu.device_count() 是 Intel stub 恒 0, 必须用 _XMLIRC
    ndev = xm._xpu_get_devices_number()
    print("  XPU 真设备数 (_XMLIRC):", ndev)
    print("  torch.cuda.device_count():", torch.cuda.device_count())

    try:
        t0 = torch.zeros(1, device="cuda:0")
        print("  cuda:0 张量 OK:", t0.device)
        print("  cuda:0 显存:", round(torch.cuda.mem_get_info(0)[1] / 1e9, 1), "GB 总量")
    except Exception as e:
        print("  [FAIL] cuda 张量创建失败:", e)
        raise SystemExit(1)

    if ndev < 2:
        print("  设备 <2，跳过 allreduce 实测（单卡部署无需跨卡）")
    else:
        world = min(ndev, 8)
        port = 24000 + os.getpid() % 1000
        import torch.multiprocessing as mp
        mp.spawn(worker, args=(world, port), nprocs=world, join=True)
