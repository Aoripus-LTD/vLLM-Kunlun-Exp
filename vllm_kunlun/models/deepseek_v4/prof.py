"""DSV4_PROF=1 分段计时（未提交调试件）。wall-clock + 全同步，仅用于相对占比。"""

import os
import time
from contextlib import contextmanager

import torch

_ENABLED = os.environ.get("DSV4_PROF") == "1"
_acc: dict[str, float] = {}
_cnt: dict[str, int] = {}
_STEP = 0
_REPORT_EVERY = 32


@contextmanager
def prof(name: str):
    if not _ENABLED:
        yield
        return
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        _acc[name] = _acc.get(name, 0.0) + (time.perf_counter() - t0)
        _cnt[name] = _cnt.get(name, 0) + 1


def step():
    global _STEP
    if not _ENABLED:
        return
    _STEP += 1
    if _STEP % _REPORT_EVERY == 0:
        report()


def report():
    total = _acc.get("__forward__", 0.0) or sum(_acc.values())
    print(f"\n[DSV4_PROF] steps={_STEP} forward_total={total:.2f}s", flush=True)
    for name, t in sorted(_acc.items(), key=lambda kv: -kv[1]):
        if name == "__forward__":
            continue
        n = _cnt[name]
        print(
            f"  {name:>24}: {t:8.2f}s  ({100.0 * t / max(total, 1e-9):5.1f}%)  n={n}  avg={1000.0 * t / max(n, 1):.2f}ms",
            flush=True,
        )
