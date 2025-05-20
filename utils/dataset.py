import torch
from torch.utils.data import Dataset


class TransliterationDataset(Dataset):
    def __init__(self, src_lines, trg_lines, src_vocab, trg_vocab, max_len=30):
        self.src_lines = src_lines
        self.trg_lines = trg_lines
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src = self.src_vocab.numericalize(self.src_lines[idx])
        trg = self.trg_vocab.numericalize(self.trg_lines[idx])
        src = src[:self.max_len] + [self.src_vocab.pad_idx] * (self.max_len - len(src))
        trg = trg[:self.max_len] + [self.trg_vocab.pad_idx] * (self.max_len - len(trg))
        return torch.tensor(src), torch.tensor(trg)
