"""
小批量随机梯度下降
"""
import numpy as np
import torch
import time
from torch import nn, optim
import sys
sys.path.append("..")
import utils

features, labels = utils.get_data_ch7()
print(features.shape) # 1500 * 5

"""
从零开始实现
"""
def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data

def train_sgd(lr, batch_size, num_epochs=2):
    utils.train_ch7(sgd, None, {'lr' : lr}, features, labels,
        batch_size, num_epochs)

train_sgd(1, 1500, 6) # 梯度下降
train_sgd(0.005, 1) # 随机梯度下降
train_sgd(0.05, 10) # 小批量随机梯度下降

"""
简洁实现
"""
utils.train_pytorch_ch7(optim.SGD, {"lr": 0.05}, features, labels, 10)