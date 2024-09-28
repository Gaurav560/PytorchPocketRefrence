import torch


#The basic slicing syntax is tensor[start_row:end_row, start_col:end_col]
x=torch.tensor([[1,2],[3,4],[5,6],[7,8]])
print(x[:2,1])