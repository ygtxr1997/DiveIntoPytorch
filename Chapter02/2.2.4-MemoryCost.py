import torch

'''
运算的内存开销
'''
# 用id可以访问变量的内存地址
# 加法 + 会开辟新内存
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before) # False

# 索引不会开辟新内存
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before) # True

# 运算符全名函数和自加运算符不会开辟新内存
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # 或者 y += x 或者 y.add_(x)
print(id(y) == id_before) # True