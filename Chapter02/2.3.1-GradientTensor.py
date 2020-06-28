import torch

# 创建tensor并设置requires_grad=True
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn) # 叶子节点, grad_fn=None

# 做运算后
y = x + 2
print(y)
print(y.grad_fn) # 中间节点, grad_fn不为空

print(x.is_leaf, y.is_leaf)

# 更复杂的运算
z = y * y * 3
out = z.mean()
print(z, out)

# 用in-place方式改变requires_grad属性
a = torch.randn(2, 2) # 默认requires_grad=False
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False
a.requires_grad_(True) # 设置requires_grad
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn) # 非空