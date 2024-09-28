import torch

#Indexing
x=torch.tensor([[1,2],[3,4],[5,6],[7,8]])
print(x)  #returns a tensor
print(x[1,1])  #indexing also returns a tensor
#.item() method returns a value as a python number
print(x[1,1].item())


#boolean indexing-> returns a tensor of the same shape as the input tensor
print(x[x>5])

