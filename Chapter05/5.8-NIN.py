"""
NIN即网络中的网络.
串联多个由卷积层+全连接层组成的子网络.
并使用1*1卷积层代替全连接层.
"""

import time
import torch
from torch import nn, optim
import sys
sys.path.append("..")
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NIN块, 由1个卷积层+2个充当全连接层的1*1卷积层串联而成
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU())
    return blk

net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),
        nin_block(384, 10, kernel_size=3, stride=1, padding=1),
        utils.GlobalAvgPool2d(),
        utils.FlattenLayer()
    )

X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)

# 训练模型
batch_size = 128
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.002, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer,
    device, num_epochs)