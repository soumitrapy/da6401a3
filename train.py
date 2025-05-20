import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from utils.dataset import TransliterationDataset
from utils.vocab import build_vocab, read_lexicon
from models.seq2seq import Seq2Seq

# Load data
data = read_lexicon("data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sample.train.tsv")
dev = read_lexicon("data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sample.dev.tsv")
x_stoi, _ = build_vocab([x for x, _ in data])
y_stoi, _ = build_vocab([y for _, y in data])

train_ds = TransliterationDataset(data, x_stoi, y_stoi)
dev_ds = TransliterationDataset(dev, x_stoi, y_stoi)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=32)

model = Seq2Seq(len(x_stoi), len(y_stoi), 64, 128, 2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, y[:, :-1])
        loss = loss_fn(out.transpose(1, 2), y[:, 1:])
        loss.backward()
        optimizer.step()
