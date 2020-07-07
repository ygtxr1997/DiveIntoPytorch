"""
并行计算
GPU是异步操作.
"""

import torch
import time
import sys
sys.path.append("..")
import utils

assert torch.cuda.device_count() >= 2

def run(x):
    for _ in range(20000):
        y = torch.mm(x, x)

x_gpu1 = torch.rand(size=(100, 100), device='cuda:0')
x_gpu2 = torch.rand(size=(100, 100), device='cuda:1')

with utils.Benchmark('Run on GPU1.'):
    run(x_gpu1)
    torch.cuda.synchronize()

with utils.Benchmark('Then run on GPU2.'):
    run(x_gpu2)
    torch.cuda.synchronize()

with utils.Benchmark('Run on both GPU1 and GPU2 in parallel.'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()