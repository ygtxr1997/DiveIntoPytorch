"""
使用重复元素的网络(VGG)
VGG块:连续使用数个相同的padding=1, size=3*3的卷积层,
      接上一个padding=2, size=2*2的最大池化层.
      并且卷积层保持尺寸不变, 池化层使得尺寸减半.
对于给定的感受野, 采用堆积的小卷积核优于采用大的卷积核,
因此在VGG中, 使用3个3*3的卷积核代替7*7的卷积核, 使用2个3*3代替1个5*5的卷积核.
"""

import time
import torch
from torch import nn, optim
import sys
sys.path.append("..")
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VGG块
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs): # 这里保持宽高不变
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里使得宽高减半
    return nn.Sequential(*blk)

# 构造一个VGG网络, 5个卷积块, 前2个使用单卷积层, 后3个使用双卷积层. 全连接层与AlexNet一样.
# 第一块输入输出通道分别为1和64, 之后每次对输出通道数翻倍, 直到变成512.
# 最终使用8个卷积层+3个全连接层, 又被称为VGG-11.
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512)) # VGG块参数序列
# 经过5个vgg_block, 宽高减半5次, 最终变成 224/32 = 7
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096

# VGG11
def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i + 1),
            vgg_block(num_convs, in_channels, out_channels))
    # 全连接部分
    net.add_module("fc", nn.Sequential(utils.FlattenLayer(),
            nn.Linear(fc_features, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, 10))
        )
    return net

net = vgg(conv_arch, fc_features, fc_hidden_units)
X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children(): # 只访问一级子模块
    X = blk(X)
    print(name, 'output shape: ', X.shape)

# 由于VGG比较复杂, 因此构建通道数相对减小
ratio = 8
small_conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio),
    (2, 128 // ratio, 256 // ratio), (2, 256 // ratio, 512 // ratio),
    (2, 512 // ratio, 512 // ratio)]
net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
print(net)

# 训练模型
batch_size = 64
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer,
    device, num_epochs)