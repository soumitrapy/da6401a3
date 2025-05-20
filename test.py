# test.py
import torch
from model import Seq2SeqModel  # Assuming you saved model architecture here
from data import load_data, prepare_dataloaders
from utils import evaluate_model, load_vocab
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_PATH = "data/hi/lexicons/test.tsv"
VOCAB_PATH = "artifacts/vocab.pt"
MODEL_PATH = "artifacts/best_model.pt"

# Load vocabularies
input_vocab, output_vocab = load_vocab(VOCAB_PATH)

# Load test data
test_pairs = load_data(TEST_PATH)
_, _, test_loader = prepare_dataloaders(test_pairs, input_vocab, output_vocab, batch_size=1)

# Load model
model = Seq2SeqModel.load(MODEL_PATH, input_vocab, output_vocab).to(DEVICE)
model.eval()

# Evaluate
acc, predictions = evaluate_model(model, test_loader, DEVICE)
print(f"Test Accuracy (Exact Match): {acc * 100:.2f}%")

import csv
os.makedirs("predictions_vanilla", exist_ok=True)

with open("predictions_vanilla/test_predictions.tsv", "w", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["Input", "Predicted", "Reference", "Correct"])
    for (inp, pred, ref) in predictions:
        writer.writerow([inp, pred, ref, pred == ref])
