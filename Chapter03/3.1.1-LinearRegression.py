# %matplotlib inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
sys.path.append("..")
from utils import *

'''
只利用tensor和autograd来实现线性回归
'''

# 生成数据集
# features: num_examples * num_inputs
# labels: num_examples * 1
def getDataset(num_inputs=2, num_examples=1000, true_w=[2, -3.4], true_b=4.2):
    features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size())) # 随机噪声
    return (features, labels)

features, labels = getDataset()
set_figsize()

# draw dataset
# plt.scatter(X[:, 0].numpy(), y.numpy(), 1)
# plt.show()

# mini-batch
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

# 初始化参数
num_inputs = 2
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float64)
b = torch.zeros(1, dtype=torch.float64)
w.requires_grad_(True)
b.requires_grad_(True)

# 正式运行
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_() # 清零
        b.grad.data.zero_() # 清零
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(w)
print(b)