import torch
import torch.nn as nn

from spnz.base import ExportImportMixin, ModeSwitcherBase
from spnz.components import Encoder


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
    ):
        super().__init__()
        self.mode = self.Mode.gradient
        self.use_pos_encode = False

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

        self.occurrence_after_transformer = nn.Sequential(
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

        gradient = (torch.cumsum(torch.ones(2 ** 14), 0).unsqueeze(1) - 1)
        self.register_buffer("gradient", gradient)

        self.multihead_attn = nn.MultiheadAttention(hidden_size, 2, batch_first=True, dropout=dropout)

    def weights_to_positions(self, weights, argmax=False, raw=False):
        batch, _trans_len, seq_len = weights.shape

        if argmax:
            return weights.max(2)[1]

        positions = (self.gradient[:seq_len] * weights.transpose(1, 2))
        if raw:
            return positions
        return positions.sum(1)

    # TODO: use pytorch embeddings
    def forward(self, features_transcription, mask_transcription, features_audio, mask_audio):
        encoder_transcription_outputs, ht = self.encoder_transcription(
            features_transcription,
            skip_pos_encode=not self.use_pos_encode,
        )

        if self.is_duration:
            return self.duration_transformer(encoder_transcription_outputs).squeeze(2)

        encoder_audio_outputs, _ = self.encoder_audio(features_audio, skip_pos_encode=not self.use_pos_encode)

        if self.is_occurrence:
            return self.occurrence_transformer(encoder_audio_outputs)

        encoder_audio_outputs = self.occurrence_after_transformer(encoder_audio_outputs)

        encoder_transcription_outputs = torch.mul(encoder_transcription_outputs, mask_transcription.unsqueeze(2))
        encoder_audio_outputs = torch.mul(encoder_audio_outputs, mask_audio.unsqueeze(2))
        attn_output, w = self.multihead_attn(
            encoder_transcription_outputs,
            encoder_audio_outputs,
            encoder_audio_outputs,
        )

        w = w * mask_audio.unsqueeze(1) * mask_transcription.unsqueeze(2)

        if self.is_weights:
            return w

        if self.is_gradient or self.is_argmax:
            return self.weights_to_positions(w, argmax=self.is_argmax)

        raise NotImplementedError(f"Mode {self.mode} not Implemented")
