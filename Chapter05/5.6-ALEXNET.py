"""
深度卷积神经网络AlexNet
包含8层变换, 其中有5层卷积, 2层全连接隐藏层, 1个全连接输出层.
使用ReLU激活函数.
Dropout正则化.
使用图像增广.
"""
import time
import torch
from torch import nn, optim
import torchvision
import sys
sys.path.append("..")
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
                nn.ReLU(),
                nn.MaxPool2d(3, 2),
                nn.Conv2d(96, 256, 5, 1, 2), # 减小卷积窗口, 填充2, 保证宽高一致, 增加输出通道
                nn.ReLU(),
                nn.MaxPool2d(3, 2),
                nn.Conv2d(256, 384, 3, 1, 1), # 进一步增加通道数
                nn.ReLU(),
                nn.Conv2d(384, 384, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(384, 256, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(3, 2)
            )
        self.fc = nn.Sequential(
                nn.Linear(256 * 5 * 5, 4096),
                nn.ReLU(),
                nn.Dropout(0.5), # 缓解过拟合
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 10) # Fashion-MNIST, 类别为10
            )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    
net = AlexNet()
print(net)

# 加载数据
batch_size = 128
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=224)

# 训练， AlexNet需要更大的显存和更长的训练时间
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer,
    device, num_epochs)