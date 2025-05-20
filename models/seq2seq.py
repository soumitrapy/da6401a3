import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hidden_dim, num_layers=1, cell_type='LSTM', dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        RNNCell = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]

        self.encoder = RNNCell(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.decoder = RNNCell(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt):
        emb_src = self.dropout(self.embedding(src))
        enc_out, hidden = self.encoder(emb_src)

        emb_tgt = self.dropout(self.embedding(tgt))
        dec_out, _ = self.decoder(emb_tgt, hidden)

        return self.fc(dec_out)
