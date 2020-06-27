import torch

'''
算术操作
'''
x = torch.rand(5, 3)
y = torch.rand(5, 3)

# add 1
print(x + y)

# add 2
print(torch.add(x, y))

# 或者指定输出
res = torch.empty(5, 3)
torch.add(x, y, out=res)
print(res)

# add 3
y.add_(x)
print(y)

'''
索引
'''