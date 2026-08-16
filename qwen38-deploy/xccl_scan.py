#!/usr/bin/env python3
# xccl_scan.py — 昆仑芯 P800 XCCL allreduce 扫描矩阵
# 扫描 world_size × 消息大小 组合，输出延迟/带宽矩阵，定位：
#   1) 2.6MB 业务点（27B TP=8 每层 allreduce 量）在带宽曲线上的位置
#   2) 8 卡拓扑是否有降速拐点（对比 world=2/4/8 的带宽走势）
# 用法: /opt/vllm_kunlun/bin/python xccl_scan.py
# 注意: 必须写成真实文件执行（mp.spawn spawn 模式需 re-import 主文件，heredoc 不行）
import os
import time

import torch
import torch.distributed as dist
import torch_xmlir._XMLIRC as xm

# 扫描矩阵：world_size 与消息大小（含 27B TP=8 业务点 ~2.6MB）
WORLDS = (2, 4, 8)
SIZES = (64 * 1024, 256 * 1024, 1024 * 1024, 2_600_000,
         4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024)
WARMUP = 5
ITERS = 50


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
        except Exception:
            pass
    if backend is None:
        if rank == 0:
            print(f"  [FAIL] world={world} 所有 backend 初始化失败")
        return
    torch.cuda.set_device(rank)
    rows = []
    for nbytes in SIZES:
        nbytes = nbytes // 4 * 4
        t = torch.randn(nbytes // 4, device=dev)
        for _ in range(WARMUP):
            dist.all_reduce(t)
        dist.barrier()
        samples = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            dist.all_reduce(t)
            samples.append(time.perf_counter() - t0)
        samples.sort()
        dt = samples[len(samples) // 2]  # 中位数抗抖动
        rows.append((nbytes, dt, nbytes / dt / 1e9))
    dist.barrier()
    if rank == 0:
        print(f"== world={world} backend={backend} ==")
        print(f"{'msg(MB)':>9} {'单轮(us)':>10} {'带宽(GB/s)':>12}")
        for nbytes, dt, bw in rows:
            print(f"{nbytes/1e6:>9.2f} {dt*1e6:>10.1f} {bw:>12.1f}")
        for nbytes, dt, bw in rows:
            if 2_500_000 <= nbytes <= 2_700_000:
                print(f"  [业务点] {world} 卡 2.6MB allreduce {dt*1e3:.2f}ms -> {bw:.1f} GB/s")
                print(f"          每 token 通信（64 层）{dt*64*1e3:.1f}ms")
                if world == 8:
                    print(f"  [TP=8 单流理论] 权重读 1.4ms + 通信 {dt*64*1e3:.1f}ms"
                          f" = {1.4+dt*64*1e3:.1f}ms/token")
    dist.destroy_process_group()


if __name__ == "__main__":
    ndev = xm._xpu_get_devices_number()
    print("XPU 真设备数 (_XMLIRC):", ndev)
    if ndev < 2:
        print("设备 <2，跳过")
        raise SystemExit(0)
    import torch.multiprocessing as mp
    for world in WORLDS:
        if world > ndev:
            continue
        port = 24000 + os.getpid() % 500 + world * 7
        mp.spawn(worker, args=(world, port), nprocs=world, join=True)
        print()
