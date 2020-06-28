'''
图像分类数据集: FASHION-MNIST, 手写数字识别数据集
torchvision包:
    torchvision.datasets        数据集接口
    torchvision.models          常用模型
    torchvision.transforms      常用图片变换
    torchvision.utils           常用工具
'''

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")
import utils as utils

# 加载数据集
mnist_train = torchvision.datasets.FashionMNIST(root='D:/Datasets/FashionMNIST',
    train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='D:/Datasets/FashionMNIST',
    train=False, download=True, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# 通过下标访问任意样本
feature, label = mnist_train[0]
print(feature.shape, label) # shape:(C, H, W)

# 查看数据集前10张图
# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# utils.show_fashion_mnist(X, utils.get_fashion_mnist_labels(y))

# 读取mini-batch
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train,
    batch_size=batch_size, shuffle=True, num_workers=num_workers) # 指定多进程加速读取, 性能瓶颈
test_iter = torch.utils.data.DataLoader(mnist_test,
    batch_size=batch_size, shuffle=False, num_workers=num_workers)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
