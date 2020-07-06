"""
稠密连接网络
和ResNet的区别是, ResNet是相加, DenseNet是通道维度上相连.
"""
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
sys.path.append("..")
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

"""
稠密块
由多个conv_block组成, 每块使用相同的输出通道数.
而卷积块的通道数控制着稠密块输出通道相对于输入通道数的增长.
"""
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X

"""
过渡层
稠密块会使得通道数增加, 用过渡层来控制模型的复杂程度.
过渡层利用1*1卷积层减小通道数, 用stride=2的平均池化层减半高和宽.
"""
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return blk

"""
DenseNet模型
单卷积, 最大池化层 -> 4个稠密块过渡层 -> 全局池化层和全连接层
"""
# 模型
net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
num_channels, growth_rate = 64, 32 # 当前通道数, 卷积层通道数
num_convs_in_dense_blocks = [4, 4, 4, 4] # 每个稠密块使用4个卷积层

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    DB = DenseBlock(num_convs, num_channels, growth_rate) # 稠密块
    net.add_module("DenseBlock_%d" % i, DB)
    num_channels = DB.out_channels # 下一个稠密块的输入通道数
    if i != len(num_convs_in_dense_blocks) - 1: # 过渡层
        net.add_module("trainsition_block_%d" % i,
            transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net.add_module("BN", nn.BatchNorm2d(num_channels))
net.add_module("relu", nn.ReLU())
net.add_module("global_avg_pool", utils.GlobalAvgPool2d())
net.add_module("fc", nn.Sequential(utils.FlattenLayer(), nn.Linear(num_channels, 10)))

# 获取数据, 训练模型
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer,
    device, num_epochs)