'''
以多项式函数拟合为例, 理解欠拟合和过拟合.
'''

import torch
import numpy as np
import sys
sys.path.append("..")
import utils

# 生成数据集
# y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + 噪声
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
    + true_w[2] * poly_features[:, 2]  + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 开始训练
num_epochs, loss = 100, torch.nn.MSELoss() # 平方损失函数

def fit_and_plot(train_features, test_features, train_labels, test_lables):
    net = torch.nn.Linear(train_features.shape[-1], 1) # Linear()模型会自动初始化参数
    
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step() # 调用库函数
        train_labels = train_labels.view(-1, 1)
        test_lables = test_lables.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_lables).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    utils.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
        range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data, '\nbias:', net.bias.data)

# 多项式拟合, 恰好
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
    labels[:n_train], labels[n_train:])

# 线性拟合, 欠拟合
fit_and_plot(features[:n_train, :], features[n_train:, :], 
    labels[:n_train], labels[n_train:])

# 训练样本不足, 过拟合
fit_and_plot(poly_features[0:10, :], poly_features[n_train:, :],
    labels[0:10], labels[n_train:])