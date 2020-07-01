"""
LENET模型(两个卷积层块 + 全连接层)
卷积层块:{卷积层(5*5, sigmoid), 最大池化层(2*2, stride=2)}
第一个卷积层输出通道:6
第二个卷积层输出通道:16
卷积层块的输出形状:(批量大小, 通道, 高, 宽)
全连接层的输出维度:平铺后, 120->84->10(最终类别)
"""

import time
import torch
from torch import nn, optim
import sys
sys.path.append("..")
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
                nn.Sigmoid(),
                nn.MaxPool2d(2, 2), # kernle_size, stride
                nn.Conv2d(6, 16, 5),
                nn.Sigmoid(),
                nn.MaxPool2d(2, 2)
            )
        self.fc = nn.Sequential(
                nn.Linear(16 * 4 * 4, 120),
                nn.Sigmoid(),
                nn.Linear(120, 84),
                nn.Sigmoid(),
                nn.Linear(84, 10)
            )
    def forward(self, img):
        feature = self.conv(img) # feature:[256, 16, 4, 4]
        output = self.fc(feature.view(img.shape[0], -1))
        return output

net = LeNet()
print(net)

# 获取数据
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size)

# 训练模型
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
utils.train_ch5(net, train_iter, test_iter, batch_size,
    optimizer, device, num_epochs)