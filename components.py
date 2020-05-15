from dependencies import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, scale=1, max_len=2048):
        super(PositionalEncoding, self).__init__()
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

        m = nn.Upsample(scale_factor=(1. / scale, 1), mode='bilinear', align_corners=True)
        shape = pe.shape
        pe = pe.view(1, 1, *shape)
        pe = m(pe).view(-1, d_model)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        # print("Dropout", self.dropout)

    def forward(self, x):
        if not self.scale:
            return x
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        x = x.transpose(0, 1)
        return self.dropout(x)


class Attention(nn.Module):
    r"""
    https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py
    Applies an attention mechanism on the output features from the decoder.
    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}
    Args:
        dim(int): The number of expected features in the output
    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    Examples::
         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        if self.dim:
            self.linear_out = nn.Linear(dim * 2, dim)

    def forward(self, output, context, mask=None):
        # https://arxiv.org/abs/1706.03762
        # context & mask is what we attend to
        batch_size, hidden_size, input_size = output.size(0), output.size(2), context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        # matrix by matrix product https://pytorch.org/docs/stable/torch.html#torch.bmm
        attn = torch.bmm(output, context.transpose(1, 2))
        # TODO: scale step missing?

        if mask is not None:
            attn.data.masked_fill_(~mask.unsqueeze(1), -float('inf'))

        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        if not self.dim:
            return attn

        mix = torch.bmm(attn, context)  # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        combined = torch.cat((mix, output), dim=2)  # concat -> (batch, out_len, 2*dim)

        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        return output, attn


class Attention(nn.Module):
    r"""
    https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py
    Applies an attention mechanism on the output features from the decoder.
    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}
    Args:
        dim(int): The number of expected features in the output
    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    Examples::
         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        if self.dim:
            self.linear_out = nn.Linear(dim * 2, dim)

    def forward(self, output, mask_output, context, mask_context):
        # https://arxiv.org/abs/1706.03762
        # context & mask is what we attend to
        batch_size, hidden_size, input_size = output.size(0), output.size(2), context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        # matrix by matrix product https://pytorch.org/docs/stable/torch.html#torch.bmm
        attn = torch.bmm(output, context.transpose(1, 2))
        # TODO: scale step missing?

        if mask_context is not None:
            if mask_output is not None:
                attn = attn.transpose(1, 2)
                attn.data.masked_fill_(~mask_output.unsqueeze(1), 0)
                attn = attn.transpose(1, 2)

            attn.data.masked_fill_(~mask_context.unsqueeze(1), -float('inf'))

        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        if not self.dim:
            return attn

        mix = torch.bmm(attn, context)  # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        combined = torch.cat((mix, output), dim=2)  # concat -> (batch, out_len, 2*dim)

        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        return output, attn


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, num_layers=2, dropout=0.1, time_scale=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(hidden_size)

        self.pos_encode = PositionalEncoding(hidden_size, dropout, scale=time_scale)

    def forward(self, previous, mask_trans, hidden_state, encoder_outputs, mask_audio):
        rnn_output, hidden_state = self.gru(previous, hidden_state)

        rnn_output = self.pos_encode(rnn_output)

        # print(rnn_output.shape if rnn_output is not None else rnn_output)
        # print(mask_trans.shape if mask_trans is not None else mask_trans)
        # print(encoder_outputs.shape if encoder_outputs is not None else encoder_outputs)
        # print(mask_audio.shape if mask_audio is not None else mask_audio)
        output, attn = self.attn(rnn_output, mask_trans, encoder_outputs, mask_audio)
        output = self.out(output)
        return output, hidden_state


class Encoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, out_dim=None, num_layers=2, dropout=0.1, time_scale=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batchnorm = nn.BatchNorm1d(embedding_size)
        # Embedding layer that will be shared with Decoder
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, out_dim or hidden_size)

        self.pos_encode = PositionalEncoding(out_dim or hidden_size, dropout, scale=time_scale)

    def forward(self, x, *ignore, skip_pos_encode=False):
        x = x.permute(0, 2, 1).contiguous()
        x = self.batchnorm(x)
        x = x.permute(0, 2, 1).contiguous()
        x, hidden = self.gru(x)
        # remove bi directional artifacts
        hidden = hidden[:2, :, :] + hidden[2:, :, :]
        x = self.fc(x)
        # x = F.tanh(x) # # # # # # #  # # # #  # # # # #  # # #  # # # # #  # # # # # TODO
        # x = torch.log1p(F.relu(x))
        if not skip_pos_encode:
            x = self.pos_encode(x)
        return x, hidden
