import torch
import torch.nn as nn

from .base import ExportImportMixin, ModeSwitcherBase
from .components import Attention, Encoder, PositionalEncoding


class SoftPointerNetwork(ModeSwitcherBase, ExportImportMixin, nn.Module):

    class Mode(ModeSwitcherBase.Mode):
        weights = "weights"
        gradient = "gradient"
        occurrence = "occurrence"
        argmax = "argmax"
        duration = "duration"

    def __init__(
        self,
        embedding_transcription_size,
        embedding_audio_size,
        hidden_size,
        dropout=0.35,
        # position encoding time scaling
        time_transcription_scale=8.344777745411855,
        time_audio_scale=1,
        position_encoding_size=32,
    ):
        super().__init__()
        self.mode = self.Mode.gradient
        self.position_encoding_size = position_encoding_size
        self.use_iter = False
        self.use_pos_encode = False
        self.use_pre_transformer = True

        self.t_transformer = nn.Sequential(
            nn.Linear(embedding_transcription_size, 32),
            nn.GELU(),
            nn.Linear(32, embedding_transcription_size),
            nn.GELU(),
        )

        self.a_transformer = nn.Sequential(
            nn.Linear(embedding_audio_size, 32),
            nn.GELU(),
            nn.Linear(32, embedding_audio_size),
            nn.GELU(),
        )

        self.duration_transformer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1),
        )

        self.occurrence_transformer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, embedding_transcription_size),
        )

        self.occurrence_pre_transformer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, hidden_size),
        )

        # self.gru = nn.GRU(
        #     hidden_size,
        #     hidden_size // 2,
        #     num_layers=2,
        #     dropout=dropout,
        #     bidirectional=True,
        #     batch_first=True,
        # )
        # self.fc = nn.Linear(hidden_size, hidden_size)
        # self.fch = nn.Linear(hidden_size, hidden_size // 2)

        self.encoder_transcription = Encoder(
            hidden_size=hidden_size,
            embedding_size=embedding_transcription_size,
            out_dim=hidden_size,
            num_layers=2,
            dropout=dropout,
            time_scale=time_transcription_scale,
        )

        self.encoder_audio = Encoder(
            hidden_size=hidden_size,
            embedding_size=embedding_audio_size,
            out_dim=hidden_size,
            num_layers=2,
            dropout=dropout,
            time_scale=time_audio_scale,
        )

        # self.attn = Attention(None)
        gradient = (torch.cumsum(torch.ones(2 ** 14), 0).unsqueeze(1) - 1)
        self.register_buffer("gradient", gradient)
        # self.pos_encode = PositionalEncoding(self.position_encoding_size, dropout, scale=time_audio_scale)

        self.multihead_attn = nn.MultiheadAttention(hidden_size, 2, batch_first=True, dropout=dropout)
        # self.multihead_attn2 = nn.MultiheadAttention(hidden_size, 2, batch_first=True)

    def weights_to_positions(self, weights, argmax=False, raw=False):
        batch, _trans_len, seq_len = weights.shape

        if argmax:
            return weights.max(2)[1]#[:, :]

        positions = (self.gradient[:seq_len] * weights.transpose(1, 2))
        if raw:
            return positions
        return positions.sum(1)#[:, :]

    # TODO: use pytorch embeddings
    def forward(self, features_transcription, mask_transcription, features_audio, mask_audio):
        # add some extra spice to inputs before encoders
        if self.use_pre_transformer:
            # TODO: move to a canonical internal size
            features_transcription = self.t_transformer(features_transcription)

        encoder_transcription_outputs, ht = self.encoder_transcription(
            features_transcription,
            skip_pos_encode=not self.use_pos_encode,
        )

        if self.is_duration:
            return self.duration_transformer(encoder_transcription_outputs).squeeze(2)

        if self.use_pre_transformer:
            features_audio = self.a_transformer(features_audio)

        encoder_audio_outputs, _ = self.encoder_audio(features_audio, skip_pos_encode=not self.use_pos_encode)

        if self.is_occurrence:
            return self.occurrence_transformer(encoder_audio_outputs)

        # clusters = encoder_audio_outputs
        # clusters = self.fc(activation(clusters))
        # encoder_audio_outputs, _ = self.gru(activation(clusters), self.fch(ht))

        # if self.is_occurrence:
        #     # clusters = encoder_audio_outputs
        #     # clusters = self.fc(activation(clusters))
        #     # clusters, _ = self.gru(activation(clusters))#, self.fch(ht))
        #     clusters = self.occurrence_transformer(clusters)
        #     return clusters
        encoder_audio_outputs = self.occurrence_pre_transformer(encoder_audio_outputs)

        encoder_transcription_outputs = torch.mul(encoder_transcription_outputs, mask_transcription.unsqueeze(2))
        encoder_audio_outputs = torch.mul(encoder_audio_outputs, mask_audio.unsqueeze(2))
        attn_output, w = self.multihead_attn(
            encoder_transcription_outputs,
            encoder_audio_outputs,
            encoder_audio_outputs,
        )
        # attn_output, w = self.multihead_attn2(attn_output, encoder_audio_outputs, encoder_audio_outputs)
        # print(w.shape)

        # w = self.attn(
        #     activation(encoder_transcription_outputs),
        #     mask_transcription,
        #     activation(encoder_audio_outputs),
        #     mask_audio,
        # )
        w = w * mask_audio.unsqueeze(1) * mask_transcription.unsqueeze(2)

        # w_cum = torch.cumsum(w, dim=2) * mask_audio.unsqueeze(1)
        #
        # mask = mask_transcription.clone()
        # mask[:, :-1] *= mask[:, 1:]
        # mask[:, -1] = False
        #
        # w_cum_roll = 1 - torch.roll(w_cum, -1, 1) * mask.unsqueeze(2)
        # w = w * w_cum_roll#.clip(0.1, 1)
        #
        # w_sum = w.clone().detach().sum(dim=2).unsqueeze(2)
        # w = torch.div(w, w_sum)
        # w = torch.nan_to_num(w, nan=0) * mask_audio.unsqueeze(1) * mask_transcription.unsqueeze(2)

        # # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        # clusters = torch.bmm(w, encoder_audio_outputs) + encoder_transcription_outputs
        # w = self.attn(
        #     activation(clusters / 2),
        #     mask_transcription,
        #     activation(encoder_audio_outputs),
        #     mask_audio,
        # )

        # if self.is_duration:
        #     clusters = attn_output + encoder_transcription_outputs
        #     return self.duration_transformer(clusters).squeeze(2)
        #
        # if self.is_occurrence:
        #     clusters = torch.bmm(w.transpose(1, 2), encoder_transcription_outputs) + encoder_audio_outputs
        #     clusters = self.fc(activation(clusters))
        #     clusters, _ = self.gru(activation(clusters), self.fch(ht))
        #     clusters = self.occurrence_transformer(clusters)
        #     return clusters

        if self.is_weights:
            return w

        if self.is_gradient or self.is_argmax:
            return self.weights_to_positions(w, argmax=self.is_argmax)

        raise NotImplementedError(f"Mode {self.mode} not Implemented")
