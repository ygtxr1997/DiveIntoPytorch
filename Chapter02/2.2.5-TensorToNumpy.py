import torch

'''
Tensor to Numpy
'''
a = torch.ones(5)
b = a.numpy() # 共享内存
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)

'''
Numpy to Tensor
'''
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a) # 共享内存
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)

c = torch.tensor(a) # 不共享内存, 拷贝
a += 1
print(a, c)