import os
import torch
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator
from utils import load_tokenizers, load_data, decode_prediction
from model_attention import Seq2SeqAttention


def test(model, data_loader, tgt_vocab, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for src, trg in data_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output.argmax(2)
            for i in range(src.size(0)):
                pred = decode_prediction(output[i], tgt_vocab)
                gold = decode_prediction(trg[i], tgt_vocab)
                predictions.append((pred, gold))
    return predictions


def write_predictions(predictions, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for pred, gold in predictions:
            f.write(f"PRED: {pred}\nGOLD: {gold}\n\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/best_attention_model.pt")
    parser.add_argument("--data_path", type=str, default="data/hi/lexicons/test.tsv")
    parser.add_argument("--output_path", type=str, default="predictions_attention/test_predictions.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizers, vocabularies, and test data
    src_tokenizer, tgt_tokenizer = load_tokenizers("hi")
    src_vocab = build_vocab_from_iterator(map(src_tokenizer, [line.split('\t')[0] for line in open(args.data_path, encoding='utf-8')]), specials=['<pad>', '<sos>', '<eos>', '<unk>'])
    tgt_vocab = build_vocab_from_iterator(map(tgt_tokenizer, [line.split('\t')[1].strip() for line in open(args.data_path, encoding='utf-8')]), specials=['<pad>', '<sos>', '<eos>', '<unk>'])

    src_vocab.set_default_index(src_vocab['<unk>'])
    tgt_vocab.set_default_index(tgt_vocab['<unk>'])

    test_loader = load_data(args.data_path, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, batch_size=32)

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    model = Seq2SeqAttention(
        input_dim=len(src_vocab),
        output_dim=len(tgt_vocab),
        emb_dim=checkpoint['emb_dim'],
        hid_dim=checkpoint['hid_dim'],
        pad_idx=tgt_vocab['<pad>'],
        dropout=0.3,
        device=device
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    predictions = test(model, test_loader, tgt_vocab, device)

    # Write to file
    write_predictions(predictions, args.output_path)
