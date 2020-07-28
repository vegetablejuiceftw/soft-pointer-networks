import torch.nn as nn

from .components import Encoder, Decoder, PositionalEncoding, LightLSTM


class MultyContextAttentionAudio(nn.Module):
    def __init__(self, embedding_transcription_size, embedding_audio_size, hidden_size, vocab_size, device,
                 dropout=0.1):
        super().__init__()
        self.device = device
        self.mode = "chain"
        self.vocab_size = vocab_size  # output size
        self.pos_encode = PositionalEncoding(vocab_size, dropout, scale=1)

        # self.encoder = Encoder(embedding_audio_size, embedding_audio_size, dropout=dropout, time_scale=None)
        self.encoder = LightLSTM(embedding_audio_size, vocab_size, with_hidden=True).to(device)
        self.fix_hidden = nn.Linear(self.encoder.hidden_size, hidden_size)
        # load(self.encoder, "/content/drive/My Drive/dataset/lstm-audio.pth")

        self.encoder_transcription = Encoder(
            hidden_size=hidden_size, embedding_size=embedding_transcription_size,
            num_layers=2, dropout=dropout, time_scale=None)

        self.decoder_transcription = Decoder(
            embedding_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size,
            num_layers=2, dropout=dropout, time_scale=None)

        self.encoder_audio = Encoder(
            hidden_size=hidden_size, embedding_size=embedding_audio_size,
            num_layers=2, dropout=dropout, time_scale=1)

        self.decoder_audio = Decoder(
            embedding_size=vocab_size, hidden_size=hidden_size, output_size=hidden_size,
            num_layers=2, dropout=dropout, time_scale=1)

        self.out = nn.Linear(hidden_size + vocab_size, vocab_size)
        self.out_chain = nn.Linear(hidden_size, vocab_size)

    def forward(self, features_transcription, mask_transcription, features_audio, mask_audio):
        features_audio = features_audio * 32768.0

        # output_size=vocab_size
        encoded_audio, hidden = self.encoder(features_transcription, mask_transcription, features_audio, mask_audio)

        if self.mode == "direct":
            return encoded_audio

        encoded_audio = self.pos_encode(encoded_audio)
        hidden = self.fix_hidden(hidden)

        # hidden_size
        encoded_transcriptions, hidden_transcription = self.encoder_transcription(features_transcription)

        # embedding_size = vocab_size,
        # hidden_size = hidden_size,
        # output_size = vocab_size,
        output_combined, _ = self.decoder_transcription(
            encoded_audio,  # vocab_size
            mask_audio,
            hidden_transcription + hidden,
            encoded_transcriptions,  # hidden_size
            mask_transcription,
        )

        if self.mode == "single":
            return output_combined

        encoded_audio_extra, hidden_audio = self.encoder_audio(features_audio)

        if self.mode == "chain":
            # embedding_size = vocab_size,
            # hidden_size = hidden_size,
            # output_size = hidden_size,
            output_audio, hidden_audio = self.decoder_audio(
                output_combined,  # vocab_size
                mask_audio,
                hidden_audio,
                encoded_audio_extra,  # hidden_size
                mask_audio)

            return self.out_chain(output_audio)

        raise NotImplementedError(f'No such mode found {self.mode}')


if __name__ == '__main__':
    multy_context_audio_model = MultyContextAttentionAudio(
        KNOWN_LABELS_COUNT, INPUT_SIZE, 256, KNOWN_LABELS_COUNT,
        device)
