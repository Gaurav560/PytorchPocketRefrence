import torch

x=torch.tensor([[1,2],[3,4],[5,6],[7,8]])
#two ways to transpose a tensor
print(x.T)  #returns a tensor
print(x.t()) #returns a tensor

#reshaping a tensor
print(x.reshape(1,8))  #returns a tensor