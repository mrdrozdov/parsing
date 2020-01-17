import torch


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        return index, item

    def __len__(self):
        return len(self.dataset)
