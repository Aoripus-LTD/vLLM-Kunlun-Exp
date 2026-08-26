import os
import sys
import time
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world, port, devs):
    dev = f"cuda:{devs[rank]}"
    try:
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://127.0.0.1:{port}",
            rank=rank, world_size=world, timeout=timedelta(seconds=20),
        )
    except Exception as e:
        print(f"  rank{rank}(cuda:{devs[rank]}) init FAIL: {str(e)[:120]}", flush=True)
        return
    t = torch.randn(1000000, device=dev)
    try:
        dist.barrier()
        for _ in range(10):
            dist.all_reduce(t)
        dist.barrier()
        if rank == 0:
            print(f"  [OK] pair cuda:{devs[0]}+cuda:{devs[1]} allreduce x10 pass", flush=True)
    except Exception as e:
        print(f"  rank{rank}(cuda:{devs[rank]}) allreduce FAIL: {str(e)[:150]}", flush=True)
    dist.destroy_process_group()


def test_pair(a, b):
    port = 29500 + a * 8 + b
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{a},{b}"
    os.environ.pop("XPU_VISIBLE_DEVICES", None)
    print(f"== pair {a}+{b} (port {port}) ==", flush=True)
    try:
        mp.spawn(worker, args=(2, port, (0, 1)), nprocs=2, join=True)
    except Exception as e:
        print(f"  spawn FAIL: {str(e)[:150]}", flush=True)


if __name__ == "__main__":
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for a, b in pairs:
        test_pair(a, b)
        time.sleep(2)
    print("done", flush=True)
