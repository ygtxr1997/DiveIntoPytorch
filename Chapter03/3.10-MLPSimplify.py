import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import utils as utils

'''
借助pytorch实现多层感知机
'''

# 定义模型
dim_inputs, dim_outputs, dim_hiddens = 784, 10, 256

net = nn.Sequential(
        utils.FlattenLayer(),
        nn.Linear(dim_inputs, dim_hiddens),
        nn.ReLU(),
        nn.Linear(dim_hiddens, dim_outputs)
    )

# 读取数据, 训练模型
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5) # 使用optim自带SGD()函数

num_epochs = 5
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs,
    batch_size, None, None, optimizer)

