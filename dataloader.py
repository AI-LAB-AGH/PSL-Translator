import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset

class DifferenceTransform:
    def __call__(self, landmarks: list) -> torch.tensor:
        landmarks = torch.tensor(np.array(landmarks))
        differences = torch.zeros([landmarks.shape[0]-1, landmarks.shape[1]])
        for frame in range(differences.shape[0]):
            differences[frame] = landmarks[frame] - landmarks[frame+1]
        return differences

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
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[self.annotations[self.samples[idx]]]

        if self.transform:
            sample = self.transform()(sample)
        else:
            sample = torch.tensor(np.array(sample), dtype=torch.float32)
        if self.target_transform:
            label = self.target_transform(label)

        return sample, label