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
# y与索引结果共享内存, 会同时修改
y = x[0, :]
y += 1
print(y)
print(x[0, :])

# 其他高级选择函数
# index_select(input, dim, index)   指定维度上选取
# masked_select(input, mask)        过滤
# non_zero(input)                   非零元素的下标
# gather(input, dim, index)         

'''
改变形状
'''
# view相当于只改变tensor的观察角度, 实际上共享内存
y = x.view(15)
z = x.view(-1, 5)
print(x.size(), y.size(), z.size())
x += 1
print(x)
print(y)

# 拷贝, 不共享内存
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)

# 将tensor转为Python number
x = torch.randn(1)
print(x)
print(x.item())

'''
线性代数
'''
# 线性代数函数
# trace                     矩阵的迹
# diag                      对角线元素
# triu/tril                 矩阵上三角/下三角
# mm/bmm                    矩阵乘法
# addmm/addbmm/addmv/addr   矩阵运算
# t                         转置
# dot/cross                 内积/外积
# inverse                   求逆
# svd                       奇异值分解