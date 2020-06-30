import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

# 矢量图显示
def use_svg_display():
    display.set_matplotlib_formats('svg')

# 设置图的尺寸
def set_figsize(figsize=(8, 6)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

# 每次返回batch_size个随机样本的特征和标签
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 随机读取样本
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)]) # 最后一次可能不足batch
        yield features.index_select(0, j), labels.index_select(0, j) # 使用yield generator, 0:按行索引

# 平方损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2 # pytorch的MSELoss没有除以2

# 小批量随机梯度下降算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # 使用data不会发生grad追踪

# 线性回归模型
def linreg(X, w, b):
    return torch.mm(X, w) + b # 矩阵乘法

# 根据数值标签, 获取fashion-mnist的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 画出多张图像和对应标签
def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy(), plt.cm.gray) # 灰度图
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

# 从mnist加载数据
def load_data_fashion_mnist(batch_size):
    # 加载数据集
    mnist_train = torchvision.datasets.FashionMNIST(root='D:/Datasets/FashionMNIST',
        train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='D:/Datasets/FashionMNIST',
        train=False, download=True, transform=transforms.ToTensor())
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train,
        batch_size=batch_size, shuffle=True, num_workers=num_workers) # 指定多进程加速读取, 性能瓶颈
    test_iter = torch.utils.data.DataLoader(mnist_test,
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return (train_iter, test_iter)

# 计算分类准确率
# 分类准确率即正确预测数量与总预测数量之比
# 使用dropout时, 对模型评估不应该使用dropout
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module): # pytorch模型
            net.eval() # 评估模式, 会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train() # 改回训练模式
        else: # 自定义模型
            if ('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() # 总正确数
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0] # 一批数据
    return acc_sum / n

# 第三章的train
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
    params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size) # 随机梯度下降
            else:
                optimizer.step() # 用pytorch自带函数时会用到
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
            % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

# 平铺多维向量
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1) # flatten

# y轴使用对数尺度, 画图函数
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
    legend=None, figsize=(10, 8)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

