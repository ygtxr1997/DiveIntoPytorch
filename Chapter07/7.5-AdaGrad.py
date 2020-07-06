"""
AdaGrad算法
根据自变量在每个维度的梯度值大小来调整各个维度上的学习率.
"""
import torch
import math
import sys
sys.path.append("..")
import utils

def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6 # g相当于导数
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
utils.show_trace_2d(f_2d, utils.train_2d(adagrad_2d))

eta = 2
utils.show_trace_2d(f_2d, utils.train_2d(adagrad_2d))

"""
从零开始实现
"""
features, labels = utils.get_data_ch7()

def init_adagrad_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float)
    s_b = torch.zeros(1, dtype=torch.float)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s.data += (p.grad.data**2)
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)

utils.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels) # 可以使用更大的学习率

"""
简洁实现
"""
utils.train_pytorch_ch7(torch.optim.Adagrad, {'lr': 0.1}, features, labels) # AdaGrad算法

"""
简洁实现
"""