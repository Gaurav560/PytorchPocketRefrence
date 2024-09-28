#in-place operations are operations that modify the contents of a tensor directly, without making a copy.
import torch

x=torch.tensor([[1,2,3,4],[4,5,6,7]])
y=torch.tensor([[9,10,11,12],[13,14,15,16]])
z=x+y
print(z)
k=-x
print(k)