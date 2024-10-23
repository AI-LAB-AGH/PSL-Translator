from torch.utils.data import Dataset
import json


class GecDataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None, val=False):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                incorrect_sentence = data["incorrect"]
                correct_sentence = data["correct"]
                self.data.append((incorrect_sentence, correct_sentence))
        boundary = int(len(self.data)*0.95)
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
