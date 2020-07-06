"""
梯度下降和随机梯度下降
"""
import numpy as np
import torch
import math
import sys
sys.path.append("..")
import utils

"""
一维梯度下降
"""
# 设f(x)=x^2
def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x
        results.append(x)
    print('epoch 10, x:', x)
    return results

res = gd(0.2)

def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    utils.set_figsize()
    utils.plt.plot(f_line, [x * x for x in f_line])
    utils.plt.plot(res, [x * x for x in res], '-o')
    utils.plt.xlabel('x')
    utils.plt.ylabel('f(x)')
    utils.plt.show()

show_trace(res)

show_trace(gd(0.01)) # 小学习率
show_trace(gd(1.1)) # 大学习率

"""
多维梯度下降
"""
eta = 0.1

def f_2d(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)

utils.show_trace_2d(f_2d, utils.train_2d(gd_2d))

"""
随机梯度下降
每次迭代随即均匀采样一个样本, 用它计算梯度来迭代x.
"""
def sgd_2d(x1, x2, s1, s2):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
        x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)

utils.show_trace_2d(f_2d, utils.train_2d(sgd_2d))
