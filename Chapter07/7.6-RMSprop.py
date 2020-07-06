"""
RMSprop算法
AdaGrad如果在迭代早期学习率降得较快且解不佳时, 在后期由于学习率过小可能难以找到有用的解.
RMSprop:
    SdW = β2 * SdW + (1 - β2) * dW^2
    Sdb = β2 * Sdb + (1 - β2) * db^2
    W := W - α * dW / sqrt(SdW + eps),
    b := b - α * db / sqrt(Sdb + eps).
"""
import torch
import math
import sys
sys.path.append("..")
import utils

"""
从零开始实现
"""
features, labels = utils.get_data_ch7()

def init_rmsprop_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float)
    s_b = torch.zeros(1, dtype=torch.float)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s.data = gamma * s.data + (1 - gamma) * (p.grad.data) ** 2
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)

utils.train_ch7(rmsprop, init_rmsprop_states(), {'lr': 0.01, 'gamma': 0.9}, features, labels)

"""
简洁实现
"""
utils.train_pytorch_ch7(torch.optim.RMSprop, {'lr': 0.01, 'alpha': 0.9}, features, labels) # RMSprop算法