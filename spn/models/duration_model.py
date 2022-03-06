import torch
import torch.nn as nn
import torch.nn.functional as fun

from .components import Decoder, Encoder


class DurationNetwork(nn.Module):

    def __init__(
        self,
        embedding_size,
        embedding_audio_size,
        hidden_size,
        vocab_size,
        device,
        ms_per_step,
        duration_scale,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.ms_per_step = ms_per_step
        self.duration_scale = duration_scale
        self.mode = "fast"

        self.encoder = Encoder(
            hidden_size=embedding_size,
            embedding_size=embedding_size,
            num_layers=2,
            dropout=dropout,
            time_scale=None,
        )

        self.encoder_transcription = Encoder(
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            num_layers=2,
            dropout=dropout,
            time_scale=None,
        )

        self.decoder_transcription = Decoder(
            embedding_size,
            hidden_size,
            hidden_size,
            num_layers=2,
            dropout=dropout,
            time_scale=None,
        )

        self.encoder_audio = Encoder(
            hidden_size=hidden_size,
            embedding_size=embedding_audio_size,
            num_layers=2,
            dropout=dropout,
            time_scale=None,
        )

        self.decoder_audio = Decoder(
            embedding_size,
            hidden_size,
            output_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            time_scale=None,
        )

        self.direct = nn.Linear(embedding_size, hidden_size)
        self.fast = nn.Linear(hidden_size * 2, hidden_size)
        self.chain = nn.Linear(hidden_size, embedding_size)
        self.shuffle = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, features_transcription, mask, features_audio, masks_audio):
        hidden_audio, encoded_audio = None, None
        hidden_transcription, encoder_transcription_extra = None, None

        # hidden_size = embedding_size
        encoded_transcription, _ = self.encoder(features_transcription)

        if self.mode not in ["direct", "audio"]:
            # hidden_size = hidden_size
            (encoder_transcription_extra, hidden_transcription) = self.encoder_transcription(features_transcription)

        if self.mode not in ["direct", "trans"]:
            # hidden_size = hidden_size
            encoded_audio, hidden_audio = self.encoder_audio(features_audio)

        output = self.direct(encoded_transcription)

        output_audio, hidden_audio = self.decoder_audio(
            encoded_transcription,
            mask,
            hidden_audio,
            encoded_audio,
            masks_audio,
        )

        if self.mode == "audio":
            output = output_audio

        elif self.mode == "trans":
            output, _ = self.decoder_transcription(
                encoded_transcription,
                mask,
                hidden_transcription,
                encoder_transcription_extra,
                mask,
            )

        elif self.mode == "fast":
            output_transcription, _ = self.decoder_transcription(
                encoded_transcription,
                mask,
                hidden_transcription,
                encoder_transcription_extra,
                mask,
            )

            output = torch.cat((output_transcription, output_audio), 2)
            output = self.fast(output)

        elif self.mode == "chain":
            # mangle the output to be acceptable for one more pass through the
            # encoders
            output_audio = self.chain(output_audio)

            # pass hidden audio as a hint
            output, hidden_transcription = self.decoder_transcription(
                output_audio,
                mask,
                hidden_audio,
                encoder_transcription_extra,
                mask,
            )

        output = torch.log1p(fun.relu(output))
        output = self.shuffle(output)
        output = fun.relu(output)
        output = self.out(output)
        output = fun.relu(output)

        total_duration = (
            masks_audio.sum(1) * self.ms_per_step if masks_audio is not None else features_audio.shape[1] *
            self.ms_per_step
        )
        if mask is not None:
            output = output * mask.unsqueeze(2).float()

        current_dur = (output.sum(1) * self.duration_scale).squeeze(1)
        current_duration_scales = total_duration / current_dur
        output = (output.squeeze(2) * current_duration_scales.unsqueeze(1)).unsqueeze(2)
        return output
