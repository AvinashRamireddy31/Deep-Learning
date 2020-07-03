import torch

# x = torch.ones(3,3, requires_grad=True)
# print(x)

# y = x + 5

# print(y)
# print(y.grad_fn)
# print(y.requires_grad)








# Gradient and backward propagation

x = torch.ones(2,2, requires_grad=True)
y = x + 3
z = y**2
res = z.mean()

print(x)
print(x.grad)

res.backward()
print(x.grad)

x = torch.ones(1,1,requires_grad = True)
print(x.requires_grad, x.grad_fn, x.is_leaf)

y = torch.ones(1,1)
print(y.requires_grad, y.grad_fn, y.is_leaf)

z = x + y
print(z.requires_grad, z.grad_fn, z.is_leaf)

p = torch.sum(x+y)
print(p.requires_grad, p.grad_fn, p.is_leaf)