"""
图像增广
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

utils.set_figsize()
img = Image.open('C:/Users/yuange/Pictures/duck.jpg')
utils.plt.imshow(img)

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    utils.show_images(Y, num_rows, num_cols, scale)

# 翻转和剪裁
apply(img, torchvision.transforms.RandomHorizontalFlip())
apply(img, torchvision.transforms.RandomVerticalFlip())
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 变化颜色
apply(img, torchvision.transforms.ColorJitter(brightness=0.5))
apply(img, torchvision.transforms.ColorJitter(hue=0.5))
apply(img, torchvision.transforms.ColorJitter(contrast=0.5))
color_aug = torchvision.transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5) # 亮度, 对比度, 饱和度, 色调
apply(img, color_aug)

# 叠加多个图像增广
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    color_aug,
    shape_aug])
apply(img, augs)