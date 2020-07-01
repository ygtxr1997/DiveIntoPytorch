"""
基于Module类构造网络模型
"""
import torch
from torch import nn

# 继承nn.Module来自定义模型
class MLP(nn.Module):
    # 重载初始化
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10) # 输出层
    # 重载向前传播
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
    # 无需定义反向传播函数, 直接调用backward即可

X = torch.rand(2, 784)
net = MLP() # 实例化
print(net)
print(net(X)) # 调用Module类的__call_()函数, 该函数会调用forward()函数

"""
MLP是一个可供自由组件的部件, 它的子类既可以是一个层又可以是一个模型
"""

# Sequential类是Module的子类,
# 现创建一个与Sequential类具有相同功能的MySequential类
class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        # 如果传入的是一个OrderedDict
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module) # 将module加入self._modules
        # 如果传入的是一些Module
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

net = MySequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
print(net)
print(net(X))

# ModuleList类是Module的子类
# 接收一个子模块的列表作为输入, 并且可以像list那样进行append和extend操作
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))
print(net[-1])
print(net)

# ModuleDict类是Module的子类
# 接受一个子模块的字典作为输入, 并且可以像字典那样进行访问操作
net = nn.ModuleDict({
        'linear': nn.Linear(784, 256),
        'act': nn.ReLU()
    })
net['output'] = nn.Linear(256, 10) # 插入一个
print(net['linear']) # 字典访问
print(net.output) # 字典访问
print(net)

"""
通过继承Module构造复杂模型FancyMLP
"""
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = torch.rand((20, 20), requires_grad=False) # 常数参数, 不训练
        self.linear = nn.Linear(20, 20)
    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1) # 使用常数参数
        x = self.linear(x) # 复用20*20的全连接层, 相当于共享参数
        # 控制流语句
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()

X = torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(X))

"""
因为FancyMLP和Sequential类都是Module的子类, 所以可以嵌套调用
"""
class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())
    def forward(self, x):
        return self.net(x)

net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP()) # 嵌套调用
X = torch.rand(2, 40)
print(net)
print(net(X))