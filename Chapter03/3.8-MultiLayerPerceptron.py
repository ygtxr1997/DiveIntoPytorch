'''
多层感知机
多个仿射变化的叠加仍然是仿射变换, 所以必须引入激活函数.
'''

import torch
import numpy as np

# ReLu
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()

# sigmoid
y = x.sigmoid()

# tanh
y = x.tanh()