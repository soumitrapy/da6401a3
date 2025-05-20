import torch
import wandb
import argparse
from torch.utils.data import DataLoader
from models.Seq2SeqAttention import Seq2SeqAttention
from utils.dataset import TransliterationDataset
from utils.vocab import Vocab
from utils.eval_metrics import compute_accuracy, decode_predictions


def load_data(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f]


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        src_lines = load_data(config.train_src)
        trg_lines = load_data(config.train_trg)
        val_src = load_data(config.val_src)
        val_trg = load_data(config.val_trg)

        src_vocab = Vocab(''.join(src_lines))
        trg_vocab = Vocab(''.join(trg_lines))

        train_data = TransliterationDataset(src_lines, trg_lines, src_vocab, trg_vocab)
        val_data = TransliterationDataset(val_src, val_trg, src_vocab, trg_vocab)
        train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config.batch_size)

        model = Seq2SeqAttention(len(src_vocab), len(trg_vocab),
                                 config.emb_dim, config.hid_dim,
                                 trg_vocab.pad_idx, config.dropout, device).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=trg_vocab.pad_idx)

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0
            for src, trg in train_loader:
                src, trg = src.to(device), trg.to(device)
                optimizer.zero_grad()
                output = model(src, trg)
                output = output[:, 1:].reshape(-1, output.shape[-1])
                trg = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            all_preds, all_trgs = [], []
            with torch.no_grad():
                for src, trg in val_loader:
                    src, trg = src.to(device), trg.to(device)
                    output = model(src, trg, 0)
                    preds = decode_predictions(output, trg_vocab)
                    all_preds.extend(preds)
                    all_trgs.extend(trg[:, 1:].tolist())

            acc = compute_accuracy(all_preds, all_trgs, trg_vocab)
            wandb.log({"val_acc": acc, "loss": total_loss / len(train_loader)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if not args.sweep:
        train()
