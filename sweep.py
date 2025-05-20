import wandb
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from models.seq2seq import Seq2Seq
from utils.dataset import TransliterationDataset
from utils.vocab import read_lexicon, build_vocab

def sweep_train():
    wandb.init()
    config = wandb.config

    data = read_lexicon("data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sample.train.tsv")
    dev = read_lexicon("data/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sample.dev.tsv")
    x_stoi, _ = build_vocab([x for x, _ in data])
    y_stoi, _ = build_vocab([y for _, y in data])

    train_ds = TransliterationDataset(data, x_stoi, y_stoi)
    dev_ds = TransliterationDataset(dev, x_stoi, y_stoi)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config.batch_size)

    model = Seq2Seq(len(x_stoi), len(y_stoi), config.emb_dim, config.hidden_dim, config.num_layers, config.cell_type, config.dropout)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, y[:, :-1])
            loss = loss_fn(output.transpose(1, 2), y[:, 1:])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        wandb.log({'loss': total_loss / len(train_loader), 'epoch': epoch})

if __name__ == "__main__":
    sweep_train()
