import torch
import torch.nn as nn

from dataloading import dto


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

    def forward(self, batch: dto.ArrayBatch):
        x, mask = batch.padded, batch.mask
        x, hidden = self.handle(x)
        x = torch.mul(x, mask.unsqueeze(-1))
        return x, hidden
