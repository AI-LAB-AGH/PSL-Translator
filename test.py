import torch

N = 2
L = 30
C = 2
H = 9
W = 16

x = torch.randn(N, L, C, H, W)
print(x[0, 0, 0, 0, 0])
x = torch.flatten(x, start_dim=0, end_dim=1)
print(x[0, 0, 0, 0])
x = torch.unflatten(x, dim=0, sizes=(N, L))
print(x[0, 0, 0, 0, 0])
