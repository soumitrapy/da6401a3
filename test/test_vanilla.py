import torch
from torch.utils.data import DataLoader
from models.Seq2Seq import Seq2Seq
from utils.dataset import TransliterationDataset
from utils.vocab import Vocab
from utils.eval_metrics import compute_accuracy, decode_predictions
import os
import json


def load_data(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f]


def test_model(model_path, test_src, test_trg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_lines = load_data(test_src)
    trg_lines = load_data(test_trg)

    src_vocab = Vocab(''.join(src_lines))
    trg_vocab = Vocab(''.join(trg_lines))

    test_data = TransliterationDataset(test_src, test_trg, src_vocab, trg_vocab)
    test_loader = DataLoader(test_data, batch_size=64)

    model = Seq2Seq(len(src_vocab), len(trg_vocab), emb_dim=64, hid_dim=64, pad_idx=trg_vocab.pad_idx, dropout=0.3, device=device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    all_preds, all_trgs = [], []
    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            preds = decode_predictions(output, trg_vocab)
            all_preds.extend(preds)
            all_trgs.extend(trg[:, 1:].tolist())

    accuracy = compute_accuracy(all_preds, all_trgs, trg_vocab)
    print(f"Test Accuracy: {accuracy:.4f}")

    os.makedirs("predictions_vanilla", exist_ok=True)
    with open("predictions_vanilla/predictions.json", "w", encoding='utf-8') as f:
        json.dump(all_preds, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    test_model(
        model_path="models/best_vanilla_model.pth",
        test_src="data/hi/lexicons/test.src",
        test_trg="data/hi/lexicons/test.trg"
    )
