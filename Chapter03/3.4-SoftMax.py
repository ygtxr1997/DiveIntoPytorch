'''
SoftMax 模型:
y1_hat, y2_hat, y3_hat = softmax(o1, o2, o3)
其中,
    yi_hat = exp(oi) / Σ(i=1, 3)exp(oi),
因此,
    y1_hat + y2_hat + y3_hat = 1
    且
    0 <= y1_hat, y2_hat, y3_hat <= 1,
即得到合法的概率分布.
'''