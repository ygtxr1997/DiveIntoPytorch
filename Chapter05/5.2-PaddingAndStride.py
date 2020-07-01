import torch
from torch import nn
import sys
sys.path.append("..")
import utils

"""
填充padding
"""
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

X = torch.rand(8, 8)
print(utils.comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
print(utils.comp_conv2d(conv2d, X).shape)

"""
步幅stride
"""
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(utils.comp_conv2d(conv2d, X).shape)
