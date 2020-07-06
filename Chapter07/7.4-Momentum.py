import torch
import sys
sys.path.append("..")
import utils

"""
梯度下降的问题
对于多维梯度下降, 可能在某个方向下降较快, 另一个方向下降较慢.
"""
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 **2

"""
动量法
VdW = βVdW + (1-β)dW # 指数加权平均
Vdb = βVdb + (1-β)db
W := W - αVdW, b := b - αVdb
"""
def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

eta, gamma = 0.4, 0.5
utils.show_trace_2d(f_2d, utils.train_2d(momentum_2d))

"""
从零开始实现
"""
features, labels = utils.get_data_ch7()

def init_momentum_states():
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float)
    v_b = torch.zeros(1, dtype=torch.float)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data

utils.train_ch7(sgd_momentum, init_momentum_states(), {'lr':0.02, 'momentum':0.5},
    features, labels)
utils.train_ch7(sgd_momentum, init_momentum_states(), {'lr':0.02, 'momentum':0.9},
    features, labels) # 不够平滑
utils.train_ch7(sgd_momentum, init_momentum_states(), {'lr':0.004, 'momentum':0.9},
    features, labels)

"""
简洁实现
"""
# 指定SGD的momentum即可实现动量法
utils.train_pytorch_ch7(torch.optim.SGD, {'lr':0.004, 'momentum':0.9}, features, labels)
