import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import utils as utils

'''
使用pytorch自带函数实现softmax回归
'''

# 获取数据
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

# 定义模型
dim_inputs = 28 * 28 * 1
dim_outputs = 10

from collections import OrderedDict
net = nn.Sequential(
        OrderedDict([
            ('flatten', utils.FlattenLayer()),
            ('linear', nn.Linear(dim_inputs, dim_outputs))
        ])
)

# 初始化参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练
num_epochs = 5
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs,
    batch_size, None, None, optimizer)