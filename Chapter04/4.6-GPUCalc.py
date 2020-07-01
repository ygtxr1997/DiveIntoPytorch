#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
from torch import nn

print(torch.cuda.is_available())


# In[4]:


print(torch.cuda.device_count())


# In[6]:


print(torch.cuda.current_device())


# In[7]:


print(torch.cuda.get_device_name(0))


# In[10]:


x = torch.tensor([1, 2, 3])
x = x.cuda(0)
print(x)
print(x.device)


# In[11]:


y = x ** 2
print(y)


# In[12]:


z = y + x.cpu()


# In[14]:


net = nn.Linear(3, 1)
print(list(net.parameters())[0].device)
net.cuda()
print(list(net.parameters())[0].device)


# In[15]:


x = torch.rand(2, 3).cuda()
print(net(x))


# In[ ]:




