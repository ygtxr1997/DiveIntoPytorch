"""
批量归一化
全连接层、卷积层、预测时的批量归一化都有所区别.
"""

import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
sys.path.append("..")
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var,
    eps, momentum):
    """
    gamma:拉伸; beta:平移; moving_mean:移动平均值; moving_var:移动方差; eps:修正值; momentum:移动;
    """
    if not is_training: # 预测模式
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else: # 训练模式
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2: # 全连接层
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else: # 二维卷积
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均值
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var +  (1.0 - momentum) * var
    Y = gamma * X_hat + beta # 拉伸和平移
    return Y, moving_mean, moving_var

class BatchNorm(nn.Module):
    """
    dim_features:对全连接层为输出个数, 对卷积层为输出通道数\n
    num_dim:总维度数
    """
    def __init__(self, dim_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2: # 全连接层
            shape = (1, dim_features)
        else: # 卷积层
            shape = (1, dim_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape)) # 加入参数列表
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
    
    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            self.training, X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9
        )
        return Y

net = nn.Sequential(
        nn.Conv2d(1, 6, 5),
        BatchNorm(6, num_dims=4),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        BatchNorm(16, num_dims=4),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),
        utils.FlattenLayer(),
        nn.Linear(16 * 4 * 4, 120),
        BatchNorm(120, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        BatchNorm(84, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(84, 10)
)

batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer,
    device, num_epochs)