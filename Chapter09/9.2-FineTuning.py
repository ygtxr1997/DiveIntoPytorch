"""
微调(迁移学习)
"""
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models # 常用预训练模型
import os
import sys
sys.path.append("..")
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    PATH = '/GPUFS/sysu_zhenghch_1/yuange/pytorch/dataset'
else:
    PATH = 'D:/Datasets'

train_imgs = ImageFolder(os.path.join(PATH, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(PATH, 'hotdog/test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
utils.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

# 使用预训练模型时, 必须和预训练做同样的处理
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize
])

# 定义和初始化模型
pretrained_net = models.resnet18(pretrained=True)
print(pretrained_net.fc)

pretrained_net.fc = nn.Linear(512, 2) # 修改为2个输出类别
print(pretrained_net.fc)

# 对fc层使用较大学习率, 其他层使用较小学习率
output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
optimizer = optim.SGD([
    {'params': feature_params},
    {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
    lr=lr, weight_decay=0.001)

# 微调模型
# 与全随机化模型相比, 微调的模型再相同迭代周期下能取得更高的精度
def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(PATH, 'hotdog/train'),
        transform=train_augs), batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(PATH, 'hotdog/test'),
        transform=test_augs), batch_size)
    loss = torch.nn.CrossEntropyLoss()
    utils.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

train_fine_tuning(pretrained_net, optimizer)