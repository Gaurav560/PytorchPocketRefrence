import torch

# Create a tensor with requires_grad=True, so PyTorch tracks its gradients
x = torch.tensor(2.0, requires_grad=True)

# Perform a simple operation: y = x^2
y = x**2

# Perform another operation: z = y * 3
z = 3 * y

# Now, we want to compute the gradients of z with respect to x
z.backward()

# After calling backward(), the gradients are stored in x.grad
print(x.grad)  # Output will be the derivative of z w.r.t x, which is 12.



