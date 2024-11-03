from torch.utils.data import Dataset


class TranslationDataset(Dataset):

    def __init__(self, file_paths, reverse=False, transform=None, target_transform=None):
        def read_from_file(file_path):
            with open(file_path, "r", encoding="UTF-8") as f:
                for i, line in enumerate(f):
                    if i % 2 == 0:
                        self.data.append([line.strip()])
                    else:
                        self.data[-1].append(line.strip())

        self.data = []
        self.back = reverse
        self.transform = transform
        self.target_transform = target_transform

        if type(file_paths) is str:
            read_from_file(file_paths)
        elif type(file_paths) in (list, tuple):
            for file_path in file_paths:
                read_from_file(file_path)

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
