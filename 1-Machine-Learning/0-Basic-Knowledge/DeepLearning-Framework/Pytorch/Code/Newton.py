import torch

def f(x):
    return x.sin()-x.exp()

def f_prime(x,f):
    return torch.autograd.grad(f(x),x,create_graph= True)

x1 = torch.tensor([10.],requires_grad = True)

for i in range(100):
    f_val = f(x1)
    fp = f_prime(x1,f)[0]    
    x1 = x1 - f_val/fp

print('solution:',x1)
print('final function value:',f(x1))