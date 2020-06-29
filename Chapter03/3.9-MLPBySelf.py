import torch
import numpy as np
import sys
sys.path.append("..")
import utils as utils

'''
手动实现多层感知机
'''

# 获取数据
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

# 定义参数
dim_inputs, dim_outputs, dim_hiddens = 784, 10, 256 # 隐藏单元设置为256个

W1 = torch.tensor(np.random.normal(0, 0.01, (dim_inputs, dim_hiddens)), dtype=torch.float)
b1 = torch.zeros(dim_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (dim_hiddens, dim_outputs)), dtype=torch.float)
b2 = torch.zeros(dim_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(True)

# 自定义激活函数ReLU
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

# 定义模型, 单隐藏层
def net(X):
    X = X.view((-1, dim_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

# 交叉熵损失函数
loss = torch.nn.CrossEntropyLoss()

# 训练模型
num_epochs, lr = 5, 100.0 # sgd函数中, lr会除以batch_size
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs,
    batch_size, params, lr)
