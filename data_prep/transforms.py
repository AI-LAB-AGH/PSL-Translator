import torch
import numpy as np


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


class ComputeDistSource:
    def __call__(self, sample: list) -> torch.tensor:
        sample = torch.tensor(np.array(sample), dtype=torch.float32)
        sample = torch.reshape(sample, (sample.shape[0], 42, 3))

        source = torch.tensor(np.array(
            [list(sample[0][0]) for landmark in range(21)] + [list(sample[0][21]) for landmark in
                                                              range(21)]))

        for frame in range(sample.shape[0]):
            sample[frame] -= source

        sample = torch.reshape(sample, (sample.shape[0], 42 * 3))
        return sample
