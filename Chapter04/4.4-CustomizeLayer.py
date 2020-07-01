"""
自定义层
"""
import torch
from torch import nn

# 自定义一个将输入减去平均值的层, 不含参数
class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()
out = layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
print(out)

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer()) # 构造更复杂的模型
y = net(torch.rand(4, 8))
print(y.mean().item())

# 定义含参数的层, 利用Parameter、ParameterList、ParameterDict
# 使用ParameterList
class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))
    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x

net = MyListDense()
print(net)

# 使用ParameterDict
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # update新增参数
    def forward(self, x, choice='linear1'): # 选择不同的参数
        return torch.mm(x, self.params[choice])
    
net = MyDictDense()
print(net)

x = torch.ones(1, 4)
print(net(x, 'linear1'))
print(net(x, 'linear2'))
print(net(x, 'linear3'))

# 嵌套使用自定义层
net = nn.Sequential(
        MyDictDense(),
        MyListDense()
    )
print(net)
print(net(x))