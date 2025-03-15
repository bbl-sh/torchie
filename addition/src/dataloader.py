import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class AdditionDataset(Dataset):
    def __init__(self, path, labels):
        self.data = pd.read_csv(path);
        self.x = self.data.drop(labels, axis = "columns").values
        self.y = self.data[labels].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

def load_dataloaders(path, label, batch_size):
    dataset = AdditionDataset(path = path, labels = label)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    return dataloader
