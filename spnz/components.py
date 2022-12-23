import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, scale=1, max_len=2048):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)
        if not self.scale:
            return

        max_len = int(max_len * scale)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # todo: The default behavior for interpolate/up sample with float
        # scale_factor will change in 1.6.0
        m = nn.Upsample(scale_factor=(1.0 / scale, 1), mode="bilinear", align_corners=True)

        shape = pe.shape
        pe = pe.view(1, 1, *shape)
        pe = m(pe).view(-1, d_model)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if not self.scale:
            return x
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        x = x.transpose(0, 1)
        return self.dropout(x)




class Encoder(nn.Module):

    def __init__(
        self,
        hidden_size,
        embedding_size,
        out_dim=None,
        num_layers=2,
        dropout=0.4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batchnorm = nn.BatchNorm1d(embedding_size)
        # Embedding layer that will be shared with Decoder
        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * num_layers, out_dim or hidden_size)

    def handle(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.batchnorm(x)
        x = x.permute(0, 2, 1).contiguous()
        x, hidden = self.gru(x)
        x = self.fc(x)
        return x, hidden

    def forward(self, batch: UtteranceBatch):
        x, mask = batch.padded, batch.masks
        x, hidden = self.handle(x)
        x = torch.mul(x, mask.unsqueeze(-1))
        return x, hidden
