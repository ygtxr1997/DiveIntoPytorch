'''
权重衰减pytorch实现
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

# 训练
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = utils.linreg, utils.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot_pytorch(wd):
    net = nn.Linear(dim_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd) # W权重衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            l.backward()
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    utils.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
        range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w', net.weight.data.norm().item())

fit_and_plot_pytorch(0)
fit_and_plot_pytorch(3)