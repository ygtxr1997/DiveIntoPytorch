"""
多GPU计算
"""

import torch
net = torch.nn.Linear(10, 1).cuda()

# 默认使用所有的GPU并行计算
net = torch.nn.DataParallel(net) # 还可指定device_ids

# 保存和加载模型, DataParallel会包上一层, 注意对应
torch.save(net.module.state_dict(), "./8.4-model.pt")

new_net = torch.nn.Linear(10, 1)
new_net.load_state_dict(torch.load("./8.4-model.pt"))