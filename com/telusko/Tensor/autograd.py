import torch
from numpy import dtype

x=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float,requires_grad=True)

y=x.pow(2).sum()
print(y)
y.backward()
print(x.grad)