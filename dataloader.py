import os
import torch
from torch.utils.data import Dataset

class RTMPDataset(Dataset):
    def __init__(self, root_dir: str):
        self.filepath = os.path.join(root_dir, 'data.pth')
        self.data = torch.load(self.filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (label, sample) = self.data[idx]
        return sample, label