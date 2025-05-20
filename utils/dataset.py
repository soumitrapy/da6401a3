import torch
from torch.utils.data import Dataset

class TransliterationDataset(Dataset):
    def __init__(self, data, x_stoi, y_stoi, max_len=30):
        self.data = data
        self.x_stoi = x_stoi
        self.y_stoi = y_stoi
        self.max_len = max_len

    def encode(self, seq, stoi):
        return [stoi[c] for c in seq]

    def pad(self, seq):
        return seq + [0] * (self.max_len - len(seq))

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(self.pad(self.encode(x, self.x_stoi))), torch.tensor(self.pad(self.encode(y, self.y_stoi)))

    def __len__(self):
        return len(self.data)
