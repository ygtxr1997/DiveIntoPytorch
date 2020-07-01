"""
读取和存储tensor
"""
import torch
from torch import nn

x = torch.ones(3)
torch.save(x, 'x.pt') # 保存tensor

x2 = torch.load('x.pt') # 读取tensor
print(x2)

y = torch.zeros(4)
torch.save([x, y], 'xy.pt') # 保存tensor列表
xy_list = torch.load('xy.pt')
print(xy_list)

torch.save({'x': x, 'y': y}, 'xy_dict.pt') # 保存tensor字典
xy = torch.load('xy_dict.pt')
print(xy)

"""
读写模型
"""
# state_dict是从参数名称映射到参数tensor的字典对象
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print(net.state_dict()) # 只有具有可学习参数的层才有条目

# 优化器optim也有state_dict
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())

# 保存和加载模型
# 1. 保存和加载模型参数
# torch.save(model.state_dict(), PATH)
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
X = torch.randn(2, 3)
Y = net(X)
PATH = './net.pt'
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
print(Y2 == Y) # true

# 2. 保存和加载整个模型
# torch.save(model, PATH)
# model = torch.load(PATH)
