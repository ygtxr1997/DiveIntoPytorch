import torch

# 创建tensor
x = torch.empty(5, 3)
print(x)

# 创建随机初始化tensor
x = torch.rand(5, 3)
print(x)

# 设置type为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 用数组创建tensor
x = torch.tensor([5.5, 3])
print(x)

# 用现有tensor创建新tensor
x = x.new_ones(5, 3, dtype=torch.float64)
print(x)

# 用现有tensor正态分布
x = torch.randn_like(x, dtype=torch.float)
print(x)