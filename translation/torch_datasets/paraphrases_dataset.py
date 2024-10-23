from datasets import load_dataset
from torch.utils.data import Dataset


class ParaphraseDataset(Dataset):
    def __init__(self, transform=None, target_transform=None, val=False):
        self.dataset = load_dataset("sdadas/ppc")
        self.data = []
        for split in self.dataset:
            for el in self.dataset[split]:
                if el["label"] == 1: # we want only exact paraphrases - https://huggingface.co/datasets/sdadas/ppc
                    self.data.append((el["sentence_A"], el["sentence_B"]))
        boundary = int(len(self.data) * 0.95)
        if val:
            self.data = self.data[boundary:]
        else:
            self.data = self.data[:boundary]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, target = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        return sample, target



