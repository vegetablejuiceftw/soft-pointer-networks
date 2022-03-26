import torch
import torch.nn as nn
import torch.nn.functional as f

from .base import ExportImportMixin, ModeSwitcherBase
from .components import Attention, Encoder, PositionalEncoding


class SoftPointerNetwork(ModeSwitcherBase, ExportImportMixin, nn.Module):

    class Mode(ModeSwitcherBase.Mode):
        weights = "weights"
        position = "position"
        gradient = "gradient"
        argmax = "argmax"

    def __init__(
        self,
        embedding_transcription_size,
        embedding_audio_size,
        hidden_size,
        device,
        dropout=0.35,
        # position encoding time scaling
        time_transcription_scale=8.344777745411855,
        time_audio_scale=1,
        position_encoding_size=32,
    ):
        super().__init__()
        self.mode = self.Mode.gradient
        self.position_encoding_size = position_encoding_size
        self.device = device
        self.use_iter = False
        self.use_pos_encode = True
        self.use_pre_transformer = True

        self.t_transformer = nn.Sequential(
            nn.Linear(embedding_transcription_size, 32),
            nn.Sigmoid(),
            nn.Linear(32, embedding_transcription_size),
            nn.Sigmoid(),
        ).to(device)

        self.a_transformer = nn.Sequential(
            nn.Linear(embedding_audio_size, 32),
            nn.Sigmoid(),
            nn.Linear(32, embedding_audio_size),
            nn.Sigmoid(),
        ).to(device)

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

        self.attn = Attention(None)
        self.gradient = (torch.cumsum(torch.ones(2 ** 14), 0).unsqueeze(1) - 1).to(device)
        self.zero = torch.zeros(hidden_size, 2048, self.position_encoding_size).to(device)
        self.pos_encode = PositionalEncoding(self.position_encoding_size, dropout, scale=time_audio_scale)

        self.to(device)

    def weights_to_positions(self, weights, argmax=False, position_encodings=False):
        batch, _trans_len, seq_len = weights.shape

        if position_encodings:
            position_encoding = self.pos_encode(
                torch.zeros(batch, seq_len, self.position_encoding_size).to(self.device),
            )
            positions = torch.bmm(weights, position_encoding)
            return positions[:, :]

        if argmax:
            return weights.max(2)[1][:, :]

        positions = (self.gradient[:seq_len] * weights.transpose(1, 2)).sum(1)[:, :]
        return positions

    def forward(self, features_transcription, mask_transcription, features_audio, mask_audio):
        # TODO: use pytorch embeddings
        batch_size, out_seq_len, _ = features_transcription.shape
        audio_seq_len = features_audio.shape[1]

        # add some temporal info for transcriptions
        features_transcription = features_transcription.clone()
        features_transcription[:, :-1] += features_transcription[:, 1:] * 0.55

        # add some extra spice to inputs before encoders
        if self.use_pre_transformer:
            # TODO: move to a canonical internal size
            features_transcription = self.t_transformer(features_transcription)
            features_audio = self.a_transformer(features_audio)

        encoder_transcription_outputs, _ = self.encoder_transcription(
            features_transcription,
            skip_pos_encode=not self.use_pos_encode,
        )
        encoder_audio_outputs, _ = self.encoder_audio(features_audio, skip_pos_encode=not self.use_pos_encode)

        if not self.use_iter:
            encoder_transcription_outputs = f.relu(encoder_transcription_outputs)
            encoder_audio_outputs = f.relu(encoder_audio_outputs)
            # not progressive batching
            w = self.attn(
                encoder_transcription_outputs,
                mask_transcription,
                encoder_audio_outputs,
                mask_audio,
            )

        else:
            encoder_transcription_outputs = f.relu(encoder_transcription_outputs)
            encoder_audio_outputs = f.relu(encoder_audio_outputs)
            w = torch.zeros(batch_size, out_seq_len, audio_seq_len).to(self.device)  # tensor to store decoder outputs

            w_masks, w_mask, iter_mask_audio = [], None, mask_audio
            for t in range(out_seq_len):
                iter_input = encoder_transcription_outputs[:, t:(t + 1), :]
                iter_memory = encoder_audio_outputs

                if len(w_masks) > 1:
                    w_mask = w_masks[0]
                    w_mask_b = w_masks[1]

                    w_mask = torch.clamp(w_mask, min=0.0, max=1)
                    w_mask[w_mask < 0.1] = 0
                    w_mask[w_mask > 0.1] = 1

                    w_mask_b = torch.clamp(w_mask_b, min=0.0, max=1)
                    w_mask_b[w_mask_b < 0.1] = 0

                    pad = 0.00
                    a, b = torch.split(iter_memory, 128, dim=2)
                    a = a * (w_mask.unsqueeze(2) * (1 - pad) + pad)
                    b = b * (w_mask_b.unsqueeze(2) * (1 - pad) + pad)
                    iter_memory = torch.cat([a, b], dim=2)
                    iter_mask_audio = mask_audio * (w_mask > 0.1) if mask_audio is not None else w_mask > 0.1

                iter_mask_transcription = None if mask_transcription is None else mask_transcription[:, t:(t + 1)]
                w_slice = self.attn(iter_input, iter_mask_transcription, iter_memory, iter_mask_audio)

                if w_mask is not None:
                    w[:, t:(t + 1), :] = w_slice * w_mask.unsqueeze(1)
                else:
                    w[:, t:(t + 1), :] = w_slice

                # update the progressive mask
                w_mask = w_slice.squeeze(1).clone()
                w_mask = torch.cumsum(w_mask, dim=1).detach()
                w_masks.append(w_mask)
                w_masks = w_masks[-2:]

        if self.is_weights:
            return w

        if self.is_gradient or self.is_argmax:
            return self.weights_to_positions(w, argmax=self.is_argmax)

        if self.is_position:
            return self.weights_to_positions(w, position_encodings=True)

        raise NotImplementedError(f"Mode {self.mode} not Implemented")


if __name__ == "__main__":
    print(SoftPointerNetwork(54, 26, 256, device="cpu"))
