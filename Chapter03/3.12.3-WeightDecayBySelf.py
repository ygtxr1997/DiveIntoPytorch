'''
权重衰减
针对过拟合, 即L2正则化
'''

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import utils

# 生成样本
# y = 0.05 + Σ(i=1, p)(0.01*xi) + 噪声, p为特征维度
n_train, n_test, dim_inputs = 20, 100, 200 # 为了模拟过拟合, 故意将训练样本数降低
true_w, true_b = torch.ones(dim_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, dim_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 初始化参数
def init_params():
    w = torch.randn((dim_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# L2惩罚项
def l2_penalty(w):
    return (w**2).sum() / 2

# 训练
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = utils.linreg, utils.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], [] # 记录loss
    for _ in range(num_epochs):
        for X, y, in train_iter:
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            utils.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    utils.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
        range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w', w.norm().item())

fit_and_plot(lambd=0) # 无权重衰减, 过拟合
fit_and_plot(lambd=3) # 有权重衰减, 过拟合缓解