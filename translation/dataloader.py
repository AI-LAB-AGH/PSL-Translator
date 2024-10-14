from torch.utils.data import Dataset


class TranslationDataset(Dataset):

    def __init__(self, files: tuple, reverse=False, transform=None, target_transform=None):
        self.data = []
        for file_path in files:
            with open(file_path, "r", encoding="UTF-8") as f:
                for i, line in enumerate(f):
                    if i % 2 == 0:
                        self.data.append([line.strip()])
                    else:
                        self.data[-1].append(line.strip())
        self.back = reverse
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx][0]
        target = self.data[idx][1]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        if self.back:
            return target, sample
        else:
            return sample, target