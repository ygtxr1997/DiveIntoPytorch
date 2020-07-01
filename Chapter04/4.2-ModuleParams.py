"""
模型参数的访问、初始化和共享
"""

import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1)) # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()
print(Y)

"""
访问模型参数
"""
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.shape) # "层数索引.weight/bias"

# 对于Sequential构造的网络, 可以通过方括号[]来访问网络中的任意一层
for name, param in net[0].named_parameters():
    print(name, param.shape, type(param))

# 如果一个Tensor(父类)是Parameter(子类), 那么它会被自动添加到模型的参数列表里
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass

n = MyModel()
for name, param in n.named_parameters():
    print(name) # 只有weight1, 没有weight2

# Parameter是Tensor, 因此可以根据data来访问参数数值, 用grad来访问参数梯度
weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad) # None
Y.backward()
print(weight_0.grad) # 方向传播后, 梯度才不为None

"""
初始化模型参数
Pytorch的init模块里提供了多种预设的初始化方法.
"""
# 初始化权重
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)

# 初始化偏差
for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)

# 还可以使用Parameter.initialize(), 来对某个特定参数进行初始化

"""
自定义初始化方法, Xavier随机初始化
"""
def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        return tensor.normal_(maen, std)

# 令权重有一半概率初始化为0, 另一半概率初始化为[-10, -5]和[5, 10]两个区间里均匀分布的随机数
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
    elif 'bias' in name:
        param.data += 1
    print(name, param.data)

"""
共享模型参数
"""
# 在Module类的forward()函数里多次调用一个层可以共享参数;
# 如果传入Sequential的模块是同一个Module实例的话, 参数也是共享的.
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear) # 同一个实例
print(net) # 这两个线性层其实是一个对象, id相同
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)

# 反向传播计算时, 共享参数的梯度是累加的
x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad) # 单次梯度是3, 两次就是6


