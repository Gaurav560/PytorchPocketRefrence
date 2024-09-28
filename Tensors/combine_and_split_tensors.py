import torch

x=torch.tensor([[1,2],[3,4],[5,6],[7,8]])
y=torch.tensor([[9,10],[11,12],[13,14],[15,16]])

#combining tensors x and y using torch.stack()
z=torch.stack((x,y))
print(z)  #returns a tensor

#splitting tensors using torch.unbind()
#input: The input tensor you want to unbind.
#dim: The dimension along which to unbind the tensor. By default, it is 0 (which means unbind along rows) and dim=1 means unbind vertically.
# After unbinding, this dimension is removed.
a,b=torch.unbind(x,1)
print(a)  #returns a tensor
print(b)
c,d,e,f=torch.unbind(y,0)
print(d)