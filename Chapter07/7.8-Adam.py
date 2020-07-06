"""
Adam在RMSprop算法基础上额外采用动量法.
并且使用偏差修正, t较小时, 再令Vt := Vt / (1 - β^t)
"""
import torch
import math
import sys
sys.path.append("..")
import utils

features, labels = utils.get_data_ch7()

utils.train_pytorch_ch7(torch.optim.Adam, {'lr': 0.01}, features, labels)