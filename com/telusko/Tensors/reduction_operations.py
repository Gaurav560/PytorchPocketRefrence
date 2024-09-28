import torch

# Example tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Get the index of the maximum value in the flattened tensor
print(torch.argmax(x))  # Output: 5 (index in the flattened tensor)


#torch.argmax(x, dim=0) — Find max values along rows for each column
print(torch.argmax(x, dim=0))  # Output: tensor([1, 1, 1])

#torch.argmax(x, dim=1) — Find max values along columns for each row
print(torch.argmax(x, dim=1))  # Output: tensor([2, 2])