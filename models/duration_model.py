from dependencies import *


class DurationNetwork(nn.Module):
    def __init__(self, embedding_size, embedding_audio_size, hidden_size, vocab_size, device, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size

        self.encoder = Encoder(hidden_size=embedding_size, embedding_size=embedding_size, num_layers=2, dropout=dropout,
                               time_scale=None)

        self.encoder_transcription = Encoder(hidden_size=hidden_size, embedding_size=embedding_size, num_layers=2,
                                             dropout=dropout, time_scale=None)
        self.decoder_transcription = Decoder(embedding_size, hidden_size, hidden_size, num_layers=2, dropout=dropout,
                                             time_scale=None)

        self.encoder_audio = Encoder(hidden_size=hidden_size, embedding_size=embedding_audio_size, num_layers=2,
                                     dropout=dropout, time_scale=None)
        self.decoder_audio = Decoder(embedding_size, hidden_size, hidden_size, num_layers=2, dropout=dropout,
                                     time_scale=None)

        self.direct = nn.Linear(embedding_size, hidden_size)
        self.fast = nn.Linear(hidden_size * 2, hidden_size)
        self.chain = nn.Linear(hidden_size, embedding_size)
        self.shuffle = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

        self.device = device
        self.mode = "fast"
        # load(self, "/content/drive/My Drive/dataset/multy-attention-duration-derp2.pth", ignore=['pos_encode'])#, ignore='pos_encode;direct;out;shuffle'.split(';'))
        load(self, "/content/drive/My Drive/dataset/multy-attention-duration-normalized-3-[10-25].pth")
        # load(self,  "/content/drive/My Drive/dataset/duration-final-[5-25].pth")

    def forward(self, input_sequence, mask, features_audio, masks_audio):
        batch_size, out_seq_len, *features = input_sequence.shape

        encoded_inputs, hidden = self.encoder(input_sequence)

        if self.mode not in ['direct', 'audio']:
            encoder_transcription_outputs, hidden_transcription = self.encoder_transcription(input_sequence)

        if self.mode not in ['direct', 'trans']:
            encoder_audio_outputs, hidden_audio = self.encoder_audio(features_audio)

        if self.mode == "direct":
            output = self.direct(encoded_inputs)
        elif self.mode == "audio":
            output, hidden_audio = self.decoder_audio(encoded_inputs, hidden_audio, encoder_audio_outputs, masks_audio)
        elif self.mode == "trans":
            output, hidden_transcription = self.decoder_transcription(encoded_inputs, hidden_transcription,
                                                                      encoder_transcription_outputs, mask)

        elif self.mode == "fast":
            output_transcription, hidden_transcription = self.decoder_transcription(encoded_inputs, mask,
                                                                                    hidden_transcription,
                                                                                    encoder_transcription_outputs, mask)
            output_audio, hidden_audio = self.decoder_audio(encoded_inputs, mask, hidden_audio, encoder_audio_outputs,
                                                            masks_audio)
            output = torch.cat((output_transcription, output_audio), 2)
            output = self.fast(output)

        elif self.mode == "chain":
            output_audio, hidden_audio = self.decoder_audio(encoded_inputs, hidden_audio, encoder_audio_outputs,
                                                            masks_audio)
            # mangle the output to be acceptable for one more pass through the encoders
            output_audio, hidden_audio = self.chain(output_audio), hidden_audio + hidden_transcription
            # pass hidden audio as a hint
            output, hidden_transcription = self.decoder_transcription(output_audio, hidden_audio,
                                                                      encoder_transcription_outputs, mask)

        elif self.mode == "iter":
            output = torch.zeros(batch_size, out_seq_len, self.vocab_size).to(
                self.device)  # tensor to store decoder outputs
            for t in range(out_seq_len):
                decoder_input = encoded_inputs[:, t:(t + 1), :]

                output_transcription, hidden_transcription = self.decoder_transcription(decoder_input,
                                                                                        hidden_transcription,
                                                                                        encoder_transcription_outputs,
                                                                                        mask)
                output_audio, hidden_audio = self.decoder_audio(decoder_input, hidden_audio, encoder_audio_outputs,
                                                                masks_audio)

                out = torch.cat((output_transcription, output_audio), 2)
                output[:, t:(t + 1), :] = self.out(out)
            return F.relu(output)

        else:
            raise Exception("wrong mode")

        output = torch.log1p(F.relu(output))
        output = self.shuffle(output)
        output = F.relu(output)
        output = self.out(output)
        output = F.relu(output)

        total_duration = masks_audio.sum(1) * ms_per_step if masks_audio is not None else features_audio.shape[
                                                                                              1] * ms_per_step
        if mask is not None:
            output = (output * mask.unsqueeze(2).float())

        current_dur = (output.sum(1) * DURATION_SCALER).squeeze(1)
        current_duration_scalers = total_duration / (current_dur)
        output = (output.squeeze(2) * current_duration_scalers.unsqueeze(1)).unsqueeze(2)
        return output
