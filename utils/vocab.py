class Vocab:
    def __init__(self, tokens, pad_token='<pad>', sos_token='<s>', eos_token='</s>'):
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.specials = [pad_token, sos_token, eos_token]
        unique_tokens = sorted(set(tokens) | set(self.specials))
        self.stoi = {tok: i for i, tok in enumerate(unique_tokens)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.pad_idx = self.stoi[pad_token]
        self.sos_idx = self.stoi[sos_token]
        self.eos_idx = self.stoi[eos_token]

    def __len__(self):
        return len(self.stoi)

    def numericalize(self, text):
        return [self.stoi[self.sos_token]] + [self.stoi[ch] for ch in text] + [self.stoi[self.eos_token]]

    def textify(self, indices):
        return ''.join([self.itos[i] for i in indices if self.itos[i] not in self.specials])
