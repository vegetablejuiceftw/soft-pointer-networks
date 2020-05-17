from dependencies import *


class MultyContextAttention(nn.Module):
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
        # load(self,  "/content/drive/My Drive/dataset/multy-attention-duration-final-[5-25].pth")

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


if __name__ == '__main__':
    duration_combined_model = MultyContextAttention(KNOWN_LABELS_COUNT, INPUT_SIZE, 256, 1, device).to(device).eval()
    evaluate(duration_combined_model, train_eval_dataset.batch(64), loss_function=MaskedLoss(),
             train_function=duration_trainer)
    show_duration_og(duration_combined_model, test_dataset, sample_size=2000)
    torch.cuda.empty_cache()

    #  Evaluation on train - avg_loss: 0.0064 count:58 Total loss:0.3695
    """
    =====  AFTER SCALING
    [duration] DIFF abs mean: 19.59ms (0.00) min:0.00ms max:1616.81ms
        20.4%	 < 5ms		39.1%	 < 10ms	
        54.5%	 < 15ms		66.5%	 < 20ms	
        75.4%	 < 25ms		81.9%	 < 30ms	
        86.4%	 < 35ms		89.8%	 < 40ms	
        92.1%	 < 45ms		93.7%	 < 50ms	
        95.0%	 < 55ms		96.0%	 < 60ms	
        96.7%	 < 65ms		97.2%	 < 70ms	
        97.7%	 < 75ms		98.0%	 < 80ms	
        98.3%	 < 85ms		98.5%	 < 90ms	
        98.7%	 < 95ms		98.8%	 < 100ms	
        98.9%	 < 105ms		100.0%	 < 9999ms	
    20.37% 39.06% 54.51% 66.53% 75.40% 81.89% 86.43% 89.76% 92.07% 93.74% 95.00% 96.00% 96.69% 97.24% 97.65% 97.98% 98.27% 98.47% 98.67% 98.82% 
    [position] DIFF abs mean: 71.81ms (-10.80) min:0.00ms max:1616.81ms
        8.1%	 < 5ms		13.3%	 < 10ms	
        18.5%	 < 15ms		23.4%	 < 20ms	
        28.5%	 < 25ms		33.1%	 < 30ms	
        37.7%	 < 35ms		42.1%	 < 40ms	
        46.2%	 < 45ms		50.0%	 < 50ms	
        53.7%	 < 55ms		57.1%	 < 60ms	
        60.2%	 < 65ms		63.1%	 < 70ms	
        65.8%	 < 75ms		68.3%	 < 80ms	
        70.7%	 < 85ms		73.0%	 < 90ms	
        75.2%	 < 95ms		77.0%	 < 100ms	
        78.8%	 < 105ms		100.0%	 < 9999ms	
    8.06% 13.26% 18.49% 23.43% 28.47% 33.12% 37.68% 42.06% 46.24% 50.02% 53.69% 57.05% 60.17% 63.12% 65.76% 68.33% 70.73% 73.04% 75.16% 76.99%
    """

    # loss_function = MaskedSoftL1(5)
    # loss_function = MaskedMSE()
    loss_function = MaskedLoss()
    # loss_function = DivMaskedMSE(5)

    train_batch_size = 64
    if WIN_STEP < 0.010:
        train_batch_size = 16

    train(duration_combined_model, 4, train_dataset.batch(train_batch_size), [train_eval_dataset.batch(64)],
          loss_function=loss_function, train_function=duration_trainer, lr_decay=0.98, lr=0.000015, weight_decay=1e-05)

    # export_model(duration_combined_model, "/content/drive/My Drive/dataset/multy-attention-duration-normalized-3-[10-25].pth")
    show_duration_og(duration_combined_model, test_dataset, sample_size=2000)
    duration_combined_model.eval()
    torch.cuda.empty_cache()

    # draw_duration(duration_combined_model, test_dataset, 36)
    # export_model(duration_combined_model, "/content/drive/My Drive/dataset/multy-attention-duration-final-[5-25].pth")
