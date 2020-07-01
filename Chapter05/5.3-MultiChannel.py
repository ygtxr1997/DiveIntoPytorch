import torch
from torch import nn
import sys
sys.path.append("..")
import utils

"""
多通道输入与多通道输出
"""
# 多输入通道2D卷积
def corr2d_multi_in(X, K):
    res = utils.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += utils.corr2d(X[i, :, :], K[i, :, :])
    return res

X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(corr2d_multi_in(X, K))

# 多输出通道2d卷积
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K]) # 用stack叠加

K = torch.stack([K, K + 1, K + 2])
print(K.shape)
print(corr2d_multi_in_out(X, K))

"""
1*1卷积层, 实际上与全连接层等价
"""
# 1*1卷积
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X) # 直接用矩阵乘法即可实现
    return Y.view(c_o, h, w)

X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print((Y1 - Y2).norm().item() < 1e-6) # True, 两种运算等价