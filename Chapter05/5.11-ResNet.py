import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
sys.path.append("..")
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
残差块:2个相同输出通道数的3*3卷积层,
       每个卷积层后接一个批量归一化层和ReLU.
       然后, 将输入跳过这两个卷积运算直接加在最后的ReLU之前.
       并且可以通过一个1*1卷积层改变通道数.
"""
blk = utils.Residual(3, 3)
X = torch.rand((4, 3, 6, 6))
print(blk(X).shape)

"""
ResNet模型:
模块使用4个残差块, 第一个模块的通道数同输入通道数一致,
之后的每个模块在第一个残差块里将上一个模块的通道数翻倍, 宽高减半.
通过跨层的数据通道, 能够训练出有效的深度神经网络.
"""
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(utils.Residual(in_channels, out_channels, use_1x1=True, stride=2))
        else:
            blk.append(utils.Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resent_block4", resnet_block(256, 512, 2))

net.add_module("global_avg_pool", utils.GlobalAvgPool2d())
net.add_module("fc", nn.Sequential(utils.FlattenLayer(), nn.Linear(512, 10)))

# 获取数据, 训练模型
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)