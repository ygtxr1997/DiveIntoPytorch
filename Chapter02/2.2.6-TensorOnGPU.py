import torch

'''
将tensor在CPU和GPU之间移动
'''
if torch.cuda.is_available():
    print("GPU OK!")
    device = torch.device("cuda") # GPU
    y = torch.ones_like(x, device=device) # 创建在GPU上的tensor
    x = x.to(device) # 移动到GPU
    z = x + y
    print(z)
    print(z.to("cpu", torch.double)) # 同时更改数据类型
else:
    print("GPU not OK!")