import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

# out是标量, 所以调用backward()时不需要指定求导变量
out.backward() # 相当于 out.backward(torch.tensor(1.))
print(x.grad)

# grad在反向传播过程是累加的, 所以反向传播之前一般需要把梯度清零
out2 = x.sum()
out2.backward()
print(x.grad) # 累加, 结果是4.5

out3 = x.sum()
x.grad.data.zero_() # 清零
out3.backward()
print(x.grad)

# 对于非标量求导, 需要指定求导变量, 求导变量与被求导变量同型
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)

v = torch.tensor([[1.0, 0.1], [0.01, 0.101]], dtype=torch.float)
z.backward(v) # 通过加权求和, 把向量转为标量
print(x.grad)

# 中断梯度追踪
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad(): # 中断梯度追踪
    y2 = x ** 3
y3 = y1 + y2

print(x.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True

y3.backward()
print(x.grad) # 结果是2

# 使用tensor.data修改tensor的值, 但是不加入计算图
x = torch.ones(1, requires_grad=True)
print(x.data)
print(x.data.requires_grad) # tensor.data独立于计算图之外

y = 2 * x
x.data *= 100 # 改变了tensor x的值, 但是这次计算没有加入计算图

y.backward()
print(x)
print(x.grad)