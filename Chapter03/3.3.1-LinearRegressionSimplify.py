import torch
import numpy as np

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

import torch.utils.data as Data # torch 读取数据

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True) # 迭代器

# nn.Module
import torch.nn as nn
class LinearNet(nn.Module): # 继承nn.Module
    def __init__(self, n_features):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net) # 打印网络结构

# nn.Sequential
# 网络层按照Sequential的顺序依次添加到计算图中
# 写法一
net = nn.Sequential(
        nn.Linear(num_inputs, 1)
        # 此处还可传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
        ('linear', nn.Linear(num_inputs, 1))
        # ......
    ]))

print(net)
print(net[0])

# 查看模型的所有可学习参数
for param in net.parameters():
    print(param)

# 初始化模型参数
from torch.nn import init

init.normal_(net[0].weight, mean=0, std=0.01) # 正态分布
init.constant_(net[0].bias, val=0) # 常量

# 损失函数
loss = nn.MSELoss()

# 优化算法
import torch.optim as optim # 包含多种优化算法

optimizer = optim.SGD(net.parameters(), lr=0.03) # 小批量随机梯度下降, 用SGD可指定不同子网使用不同的学习率
print(optimizer)

# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零
        l.backward()
        optimizer.step() # 运行一步
    print('epoch %d, loss: %f' % (epoch, l.item()))

# 比较结果
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)