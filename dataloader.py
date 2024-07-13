import os
import csv
import torch
import numpy as np
import mediapipe as mp
from skimage import io
from torch.utils.data import Dataset


class ComputeDistConsec:
    def __call__(self, sample: list) -> torch.tensor:
        sample = torch.tensor(np.array(sample))
        differences = torch.zeros([sample.shape[0] - 1, sample.shape[1]])
        for frame in range(differences.shape[0]):
            differences[frame] = sample[frame] - sample[frame + 1]
        return differences


class ComputeDistFirst:
    def __call__(self, sample: list) -> torch.tensor:
        sample = torch.tensor(np.array(sample))
        differences = torch.zeros([sample.shape[0] - 1, sample.shape[1]])
        for frame in range(differences.shape[0]):
            differences[frame] = sample[frame + 1] - sample[0]
        return differences


class ExtractLandmarks:
    def __init__(self, holistic):
        self.holistic = holistic

    def __call__(self, sample: list) -> list:
        # Landmarks already extracted
        if len(sample[0].shape) != 3:
            return sample

        processed = []
        for frame in sample:
            results = self.holistic.process(frame)
            keypoints = np.array([])
            for landmark_list in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if landmark_list is not None:
                    for landmark in landmark_list.landmark:
                        keypoints = np.append(keypoints, [landmark.x, landmark.y, landmark.z])
                else:
                    keypoints = np.append(keypoints, np.zeros(21 * 3))
            processed.append(keypoints.copy())
        return processed


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
            self.data[idx] = [io.imread(os.path.join(path, frame)) for frame in frames]

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
