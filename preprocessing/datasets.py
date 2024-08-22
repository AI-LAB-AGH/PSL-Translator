import os
import csv
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset
#from preprocessing.transforms import ComputeDistances


class LandmarksDataset(Dataset):
    def __init__(self, root_dir: str, annotations: str, labels_map: dict, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.samples = {idx: dirname for idx, dirname in enumerate(os.listdir(root_dir))}
        with open(annotations, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            self.annotations = {row[0]: row[1] for row in reader}
        self.labels = labels_map
        self.transform = transform
        self.target_transform = target_transform

        self.data = {}
        for idx, dirname in self.samples.items():
            path = os.path.join(self.root_dir, dirname)
            frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
            self.data[idx] = [np.load(os.path.join(path, frame)) for frame in frames]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[self.annotations[self.samples[idx]]]

        if self.transform:
            sample = self.transform(sample)
        else:
            sample = torch.tensor(np.array(sample), dtype=torch.float32)
        if self.target_transform:
            label = self.target_transform(label)

        return sample, label


class JesterDataset(Dataset):
    def __init__(self, root_dir: str, annotations: str, labels_map: dict, transform=None, target_transform=None, max_samples=200):
        self.root_dir = root_dir
        self.samples = {idx: dirname for idx, dirname in enumerate(os.listdir(root_dir))}
        with open(annotations, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            self.annotations = {row[0]: row[1] for row in reader}
        self.labels = labels_map
        self.transform = transform
        self.target_transform = target_transform

        self.data = {}
        for i, (idx, dirname) in enumerate(self.samples.items()):
            path = os.path.join(self.root_dir, dirname)
            frames = sorted(os.listdir(path), key=lambda a: int(os.path.splitext(a)[0]))
            self.data[idx] = [io.imread(os.path.join(path, frame)) for frame in frames]
            print(f'\rLoaded sample {i} of {max_samples}', end='')
            if len(self.data) == max_samples:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[self.annotations[self.samples[idx]]]

        if self.transform:
            sample = self.transform(sample)
        else:
            sample = torch.tensor(np.array(sample), dtype=torch.float32)
        if self.target_transform:
            label = self.target_transform(label)

        return sample, label

class RTMPDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, target_transform=None):
        self.filepath = os.path.join(root_dir, 'data.pth')
        self.data = torch.load(self.filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (label, sample) = self.data[idx]
        return sample, label


class OFDataset(Dataset):
    def __init__(self, root_dir: str):
        self.data = None
        self.curr_first = 0
        self.root_dir = root_dir
        self.num_samples = self.count_samples()
        self.batch_size = min(50, self.num_samples)
        self.load_data(0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.curr_first <= idx < self.curr_first + self.batch_size:
            (label, sample) = self.data[idx - self.curr_first]
        else:
            self.load_data(idx - (idx % self.batch_size))
            (label, sample) = self.data[idx - self.curr_first]
        return sample, label

    def load_data(self, first_idx):
        if first_idx + self.batch_size > self.num_samples:
            filename = f'data_{first_idx}_{self.num_samples - 1}.pth'
        else:
            filename = f'data_{first_idx}_{first_idx + self.batch_size - 1}.pth'
        self.data = torch.load(os.path.join(self.root_dir, filename))
        self.curr_first = first_idx
    
    def count_samples(self):
        return max(int(filename[:-4].split('_')[2]) for filename in os.listdir(self.root_dir)) + 1 # Proud of this one-liner, ngl

# ofds = OFDataset(os.path.join('data', 'RGB_OF', 'train'))
# ofds.count_samples()
