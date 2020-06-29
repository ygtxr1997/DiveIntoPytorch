import torch
import torchvision
import numpy as np
import sys
sys.path.append("..")
import utils as utils

'''
手动实现SoftMax回归
'''

# 读取数据
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

# 初始化模型参数
dim_inputs = 28 * 28 * 1 # 图片尺寸
dim_outputs = 10 # 10种分类

W = torch.tensor(np.random.normal(0, 0.01, (dim_inputs, dim_outputs)), dtype=torch.float)
b = torch.zeros(dim_outputs, dtype=torch.float)
W.requires_grad_(True)
b.requires_grad_(True)

# SoftMax
# 对于输入X, 先将每个元素变成负数, 然后除以每行的和使得每一行和为1
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition # broadcast

# 单层softmax模型
def net(X):
    # 使用view将原始输入图像展开成向量
    return softmax(torch.mm(X.view((-1, dim_inputs)), W) + b)

# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1))) # 对y_hat的每个预测值, 取y对应的那个位置对应的值

# 训练
num_epochs, lr = 5, 0.1
utils.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,
    batch_size, [W, b], lr)

# 预测
X, y = iter(test_iter).next()

true_labels = utils.get_fashion_mnist_labels(y.numpy())
pred_labels = utils.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

utils.show_fashion_mnist(X[0:9], titles[0:9])