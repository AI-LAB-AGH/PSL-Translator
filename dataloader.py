import os
import csv
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset


class ComputeDistConsec:
    def __call__(self, sample: tuple[list, list]) -> torch.tensor:
        sample = torch.tensor(np.array(sample))
        differences = torch.zeros([sample.shape[0] - 1, sample.shape[1]])
        for frame in range(differences.shape[0]):
            differences[frame] = sample[frame] - sample[frame + 1]
        return differences


class ComputeDistFirst:
    def __call__(self, sample: tuple[list, list]) -> torch.tensor:
        sample = torch.tensor(np.array(sample))
        differences = torch.zeros([sample.shape[0] - 1, sample.shape[1]])
        for frame in range(differences.shape[0]):
            differences[frame] = sample[frame + 1] - sample[0]
        return differences


class ComputeDistSource:
    def __call__(self, sample: tuple[list, list]) -> tuple[torch.tensor, torch.tensor]:
        sample = torch.tensor(np.array(sample), dtype=torch.float32)
        sample = torch.reshape(sample, (sample.shape[0], 42, 3))

        source = torch.tensor(np.array(
            [list(sample[0][0]) for landmark in range(21)] + [list(sample[0][21]) for landmark in
                                                              range(21)]))

        for frame in range(sample.shape[0]):
            sample[frame] -= source

        sample = torch.reshape(sample, (sample.shape[0], 42 * 3))
        return sample


class ComputeDistNetWithMovement:
    def __call__(self, sample: tuple[list, list]) -> tuple[torch.tensor, torch.tensor]:
        left = sample[0]
        left = torch.tensor(np.array(left), dtype=torch.float32)
        left = torch.reshape(left, (left.shape[0], 21, 3))
        source = left[0][0]
        for frame in left:
            frame[1:] -= frame[0]
            frame[0] -= source
        left = torch.reshape(left, (left.shape[0], 21 * 3))

        right = sample[1]
        right = torch.tensor(np.array(right), dtype=torch.float32)
        right = torch.reshape(right, (right.shape[0], 21, 3))
        source = right[0][0]
        for frame in left:
            frame[1:] -= frame[0]
            frame[0] -= source
        right = torch.reshape(right, (right.shape[0], 21 * 3))

        return (left, right)
    
class ComputeDistNetNoMovement:
    def __call__(self, sample: tuple[list, list]) -> tuple[torch.tensor, torch.tensor]:
        left = sample[0]
        for frame in range(len(left)):
            left[frame] = torch.tensor(np.array(left[frame]), dtype=torch.float32)
            if left[frame].shape[0] != 0:
                left[frame] = torch.reshape(left[frame], (21, 3))
                source = left[frame][0].clone()
                left[frame] -= source
                left[frame] = left[frame].view(21 * 3)

        right = sample[1]
        for frame in range(len(right)):
            right[frame] = torch.tensor(np.array(right[frame]), dtype=torch.float32)
            if right[frame].shape[0] != 0:
                right[frame] = torch.reshape(right[frame], (21, 3))
                source = right[frame][0].clone()
                right[frame] -= source
                right[frame] = right[frame].view(21 * 3)

        return (left, right)


class ExtractLandmarks:
    def __init__(self, holistic):
        self.holistic = holistic

    def __call__(self, sample: list) -> tuple[list, list]:
        # Landmarks already extracted
        if len(sample[0].shape) != 3:
            return sample

        left = []
        right = []
        for frame in sample:
            results = self.holistic.process(frame)
            keypoints = np.array([])

            if results.left_hand_landmarks is not None:
                for landmark in results.left_hand_landmarks.landmark:
                    keypoints = np.append(keypoints, [landmark.x, landmark.y, landmark.z])
            else:
                keypoints = np.append(keypoints, [])
            left.append(keypoints.copy())
            
            keypoints = np.array([])
            if results.right_hand_landmarks is not None:
                for landmark in results.right_hand_landmarks.landmark:
                    keypoints = np.append(keypoints, [landmark.x, landmark.y, landmark.z])
            else:
                keypoints = np.append(keypoints, [])
            right.append(keypoints.copy())

        return (left, right)


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


class FramesDataset(Dataset):
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
