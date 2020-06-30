'''
手动实现dropout
测试模型时一般不使用dropout.
'''

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import utils

# dropout
def dropout(X, drop_prob):
    """
    X: 矩阵; drop_prob: float, 丢弃概率
    """
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0: # 全丢
        return torch.zeros_like(X)
    mask = (torch.randn(X.shape) < keep_prob).float()

    return mask * X / keep_prob

X = torch.arange(16).view(2, 8)
print(X)
print(dropout(X, 0.5))

# 定义模型参数
dim_inputs, dim_outputs, dim_hiddens1, dim_hiddens2 = 784, 10, 256, 256

W1 = torch.tensor(np.random.normal(0, 0.01, size=(dim_inputs, dim_hiddens1)),
    dtype=torch.float, requires_grad=True)
b1 = torch.zeros(dim_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(dim_hiddens1, dim_hiddens2)),
    dtype=torch.float, requires_grad=True)
b2 = torch.zeros(dim_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(dim_hiddens2, dim_outputs)),
    dtype=torch.float, requires_grad=True)
b3 = torch.zeros(dim_outputs, requires_grad=True)

params = [W1, b1, W2, b2, W3, b3]

# 定义模型
drop_prob1, drop_prob2 = 0.2, 0.5

def net(X, is_training=True):
    """
    X: 输入; is_training: bool, 是否为训练集
    """
    X = X.view(-1, dim_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu() # relu()
    if is_training:
        H1 = dropout(H1, drop_prob1) # 第一层dropout
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2) # 第二层dropout
    return torch.matmul(H2, W3) + b3

# 训练模型
num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs,
    batch_size, params, lr)
