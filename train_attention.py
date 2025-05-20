import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import argparse
from model_attention import Seq2SeqAttention
from utils import get_datasets, save_predictions, set_seed


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    total_correct = 0
    total_count = 0

    for i, batch in enumerate(iterator):
        src, trg = batch[0].to(model.device), batch[1].to(model.device)
        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        pred_tokens = output.argmax(dim=1)
        total_correct += (pred_tokens == trg).sum().item()
        total_count += trg.numel()

    accuracy = total_correct / total_count
    return epoch_loss / len(iterator), accuracy


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch[0].to(model.device), batch[1].to(model.device)
            output = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

            pred_tokens = output.argmax(dim=1)
            total_correct += (pred_tokens == trg).sum().item()
            total_count += trg.numel()

    accuracy = total_correct / total_count
    return epoch_loss / len(iterator), accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--lang', type=str, default='hi')
    args = parser.parse_args()

    set_seed(args.seed)
    wandb.init(project="dakshina-attention", config=vars(args))

    train_loader, val_loader, input_dim, output_dim, pad_idx = get_datasets(args.lang, args.batch_size)

    model = Seq2SeqAttention(input_dim, output_dim, args.embedding_dim,
                             args.hidden_dim, pad_idx, dropout=args.dropout, device=args.device)
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, clip=1)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'epoch': epoch
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_attention_model.pth")
            print(f"Saved new best model at epoch {epoch} with val acc {val_acc:.4f}")


if __name__ == '__main__':
    main()