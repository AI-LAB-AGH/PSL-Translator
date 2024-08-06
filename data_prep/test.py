import torch
import os

path = os.path.join('data', 'RGB_OF', 'train', 'data.pth')

data = torch.load(path)

print(data[0][1][0].shape)
print(len(data))
