#!/usr/bin/env python3
"""PyTorch's single-threaded fp32 matmul on d1's projection shapes: the bar for BFloat16Matrix.

    python tools/bench_gemm.py 61 116 600
"""
import sys
import time

import torch

torch.set_num_threads(1)
SHAPES = [("conv.in_proj", 6144, 2048), ("conv.out_proj", 2048, 2048), ("attn.qkv", 3072, 2048),
          ("mlp.w1|w3", 21504, 2048), ("mlp.w2", 2048, 10752)]
rows = [int(a) for a in sys.argv[1:]] or [61, 116, 600]
for name, n, k in SHAPES:
    w = torch.randn(n, k) * 0.02
    linear = torch.nn.Linear(k, n, bias=False)
    linear.weight.data = w
    for t in rows:
        x = torch.randn(1, t, k)
        with torch.inference_mode():
            linear(x)
            repeats = max(3, int(2e10 / (2 * t * n * k)))
            start = time.perf_counter()
            for _ in range(repeats):
                linear(x)
            s = (time.perf_counter() - start) / repeats
        print(f"{name:14s} rows {t:4d}  [{n}x{k}]  {s * 1e3:7.2f} ms  {2 * t * n * k / s / 1e9:6.1f} GFLOP/s", flush=True)
