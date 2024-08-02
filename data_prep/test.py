import torch
import os

path = os.path.join('data', 'RGB_P', 'train', 'data.pth')

data = torch.load(path)

print(data[0][2][0].shape)
