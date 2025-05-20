def compute_accuracy(preds, targets, trg_vocab):
    pred_texts = [trg_vocab.textify(p) for p in preds]
    target_texts = [trg_vocab.textify(t) for t in targets]
    match_count = sum([p == t for p, t in zip(pred_texts, target_texts)])
    return match_count / len(pred_texts)


def decode_predictions(output_tensor, trg_vocab):
    preds = []
    for output in output_tensor:
        pred_tokens = output.argmax(dim=1).tolist()
        if trg_vocab.eos_idx in pred_tokens:
            pred_tokens = pred_tokens[:pred_tokens.index(trg_vocab.eos_idx)]
        preds.append(pred_tokens)
    return preds
