import torch
import torch.nn as nn


class Seq2SeqVanilla(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, pad_idx, dropout, device):
        super().__init__()
        self.device = device
        self.encoder = nn.GRU(input_dim, hid_dim, batch_first=True)
        self.decoder = nn.GRU(output_dim, hid_dim, batch_first=True)
        self.embedding_src = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.embedding_trg = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_dim = self.fc_out.out_features

        embedded_src = self.dropout(self.embedding_src(src))
        _, hidden = self.encoder(embedded_src)

        inputs = trg[:, 0]
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

        for t in range(1, trg_len):
            embedded = self.dropout(self.embedding_trg(inputs)).unsqueeze(1)
            output, hidden = self.decoder(embedded, hidden)
            prediction = self.fc_out(output.squeeze(1))
            outputs[:, t] = prediction
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            inputs = trg[:, t] if teacher_force else top1

        return outputs
