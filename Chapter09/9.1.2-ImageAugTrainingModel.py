"""
使用图像增广训练模型
通常只在训练集使用.
"""
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

import sys
sys.path.append("..")
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 下载数据集
if torch.cuda.is_available():
    PATH = '/GPUFS/sysu_zhenghch_1/yuange/pytorch/dataset/CIFAR10'
else:
    PATH = 'D:/Datasets/CIFAR10'
all_images = torchvision.datasets.CIFAR10(train=True, root=PATH, download=True)
utils.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

# 图像增广
flip_aug = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor() # 转为pytorch可用格式
])

no_aug = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 获取数据
num_workers = 0 if sys.platform.startswith('win') else 4
def load_cifar10(is_train, augs, batch_size, root=PATH):
    dataset = torchvision.datasets.CIFAR10(root=PATH,
        train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

# ResNet模型:
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
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
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

# 训练模型
def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size = 256
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    utils.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)

train_with_data_aug(flip_aug, no_aug)