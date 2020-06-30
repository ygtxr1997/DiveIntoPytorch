'''
pytorch实现dropout
net.eval()  # 评估模式, 会关闭dropout
net.train() # 改回训练模式
'''

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import utils

# 定义模型参数
dim_inputs, dim_outputs, dim_hiddens1, dim_hiddens2 = 784, 10, 256, 256
drop_prob1, drop_prob2 = 0.2, 0.5

# 定义模型
net = nn.Sequential(
        utils.FlattenLayer(),
        nn.Linear(dim_inputs, dim_hiddens1),
        nn.ReLU(),
        nn.Dropout(drop_prob1), # pytorch DropOut
        nn.Linear(dim_hiddens1, dim_hiddens2),
        nn.ReLU(),
        nn.Dropout(drop_prob2),
        nn.Linear(dim_hiddens2, 10)
    )

for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

# 训练模型
num_epochs, lr, batch_size = 5, 100.0, 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs,
    batch_size, None, None, optimizer)
