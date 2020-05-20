from time import time

from dependencies import *


class MultyContextAttentionAudio(nn.Module):
    def __init__(self, embedding_size, embedding_audio_size, hidden_size, vocab_size, device, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size

        # self.encoder = Encoder(embedding_audio_size, embedding_audio_size, num_layers=2, dropout=dropout, time_scale=None)
        self.encoder = LightLSTM(embedding_audio_size, vocab_size, with_hidden=True).to(device)
        # load(self.encoder, "/content/drive/My Drive/dataset/lstm-audio.pth")

        self.encoder_transcription = Encoder(hidden_size, embedding_size, num_layers=2, dropout=dropout,
                                             time_scale=None)
        self.decoder_transcription = Decoder(vocab_size, hidden_size, vocab_size, num_layers=2, dropout=dropout,
                                             time_scale=None)

        self.encoder_audio = Encoder(hidden_size, embedding_audio_size, num_layers=2, dropout=dropout, time_scale=1)
        self.decoder_audio = Decoder(vocab_size, hidden_size, hidden_size, num_layers=2, dropout=dropout, time_scale=1)

        self.out = nn.Linear(hidden_size * 2, vocab_size)
        self.out_single = nn.Linear(hidden_size, vocab_size)
        self.out_chain = nn.Linear(hidden_size, vocab_size)
        self.out_direct = nn.Linear(vocab_size, vocab_size)

        self.fix_hidden = nn.Linear(self.encoder.hidden_size, hidden_size)
        self.pos_encode = PositionalEncoding(vocab_size, dropout, scale=1)

        self.device = device

        self.mode = "chain"
        # load(self, "/content/drive/My Drive/dataset/multy-attention-audio-7.pth", ignore=['pos_encode'])  # without pos encode
        # load(self, "/content/drive/My Drive/dataset/multy-attention-audio-pos-6-half.pth", ignore=['pos_encode'])  # with pos encode
        load(self, "/content/drive/My Drive/dataset/CTC-final.pth")

    def forward(self, features_transcription, mask_transcription, features_audio, mask_audio):
        features_audio = features_audio * 32768.0
        batch_size, audio_len, *features = features_audio.shape
        batch_size, out_seq_len, *features = features_transcription.shape

        # if mask_transcription is None: mask_transcription = torch.ones(batch_size, out_seq_len).bool().to(device)
        # if mask_audio is None: mask_audio = torch.ones(batch_size, audio_len).bool().to(device)

        encoded_inputs, hidden = self.encoder(features_transcription, mask_transcription, features_audio, mask_audio)

        if self.mode == "direct":
            return encoded_inputs

        encoded_inputs = self.pos_encode(encoded_inputs)
        hidden = self.fix_hidden(hidden)
        encoder_transcription_outputs, hidden_transcription = self.encoder_transcription(features_transcription)

        if self.mode == "single":
            output_transcription, hidden_transcription = self.decoder_transcription(encoded_inputs, mask_audio,
                                                                                    hidden_transcription + hidden,
                                                                                    encoder_transcription_outputs,
                                                                                    mask_transcription)
            # return self.out_single(output_transcription)
            return output_transcription

        encoder_audio_outputs, hidden_audio = self.encoder_audio(features_audio)

        if self.mode == "fast":
            output_transcription, hidden_transcription = self.decoder_transcription(encoded_inputs,
                                                                                    hidden_transcription,
                                                                                    encoder_transcription_outputs,
                                                                                    mask_transcription)
            output_audio, hidden_audio = self.decoder_audio(encoded_inputs, hidden_audio, encoder_audio_outputs,
                                                            mask_audio)
            output = torch.cat((output_transcription, output_audio), 2)
            return self.out(output)

        if self.mode == "chain":
            output_transcription, hidden_transcription = self.decoder_transcription(
                # previous, mask_trans, hidden_state, encoder_outputs, mask_audio
                encoded_inputs, mask_audio, hidden_transcription + hidden, encoder_transcription_outputs,
                mask_transcription,
            )
            # encoded_inputs = encoded_inputs + self.out_single(output_transcription)
            # hidden_audio = hidden_audio + hidden_transcription + hidden
            output_audio, hidden_audio = self.decoder_audio(output_transcription, mask_audio, hidden_audio,
                                                            encoder_audio_outputs, mask_audio)
            return self.out_chain(output_audio)

            # # elif self.mode == "iter":
        # #     outputs = torch.zeros(batch_size, out_seq_len, self.vocab_size).to(self.device) # tensor to store decoder outputs
        # #     output = outputs[:,:1].clone()

        # #     for t in range(out_seq_len):
        # #         decoder_input = encoded_inputs[:,t:(t+1),:]

        # #         output_transcription, hidden_transcription = self.decoder_transcription(decoder_input, hidden_transcription, encoder_transcription_outputs, mask)
        # #         output_audio, hidden_audio = self.decoder_audio(decoder_input, hidden_audio, encoder_audio_outputs, masks_audio)

        # #         output = torch.cat((output_transcription, output_audio), 2)
        # #         output = self.out(output)
        # #         outputs[:,t:(t+1),:] = output
        # else:
        #     raise Exception("wrong mode")

        return outputs
