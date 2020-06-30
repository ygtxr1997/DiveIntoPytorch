"""
数值衰减和数值爆炸
随机初始化模型参数: torch.nn.init.normal_(), pytorch中nn.module的模块参数都采取比较合理的初始化策略
Xavier随机初始化:
    设输入维度为a, 输出维度为b, 则使得该层权重参数的每个元素都随机采样于均匀分布
    U(-sqrt(6/(a+b)), sqrt(6/(a+b))).
"""

