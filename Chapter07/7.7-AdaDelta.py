import torch
import math
import sys
sys.path.append("..")
import utils

# AdaDelta没有学习率超参数

features, labels = utils.get_data_ch7()
utils.train_pytorch_ch7(torch.optim.Adadelta, {'rho': 0.9}, features, labels)