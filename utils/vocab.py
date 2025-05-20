def build_vocab(sequences):
    chars = sorted(set(char for seq in sequences for char in seq))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi['<pad>'] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def read_lexicon(path, max_len=None):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split('\t')[:2] for line in f]
    return [(r, n) for r, n in lines if (max_len is None or (len(r) <= max_len and len(n) <= max_len))]
