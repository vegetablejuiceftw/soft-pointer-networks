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


multy_context_audio_model = MultyContextAttentionAudio(KNOWN_LABELS_COUNT, INPUT_SIZE, 256, KNOWN_LABELS_COUNT,
                                                       device).to(device);
# toy_dataset = DirectMaskDataset('train', limit=127)
# train_iter = BucketIterator(train_dataset, batch_size=64, sort_key=lambda x: len(x.features), sort=False, shuffle=True, sort_within_batch=True)
# evaluate_audio("train", train_iter, multy_context_audio_model)
#  Evaluation on train - avg_loss: 0.158870 count:72 Total loss:11.438619658350945
show_audio(multy_context_audio_model, test_dataset, "test", plot_only=True)

# duration_model = duration_combined_model
torch.cuda.empty_cache()
work_dataset = train_dataset
evaluation = [test_dataset.batch(64), train_eval_dataset.batch(64)]

for i in range(1):
    start = time()
    train(multy_context_audio_model, 8, work_dataset.batch(16), evaluation,
          loss_function=LabelSmoothingLossAudio(KNOWN_LABELS_COUNT), train_function=audio_detection_trainer,
          lr_decay=0.98, lr=0.000161 * 0.131 * 3.6)
    print(time() - start)

"""
[DETECTION+DTW]AVERAGE ERROR: 8.67% COUNT:1338
[DETECTION]AVERAGE ERROR: 9.99% COUNT:1338
[test] DIFF abs mean: 7.25ms (0.14) min:0.02ms max:278.61ms
	79.5%	 < 10ms
	93.1%	 < 20ms
	96.8%	 < 30ms
	98.3%	 < 40ms
	99.0%	 < 50ms
	99.5%	 < 65ms
	99.8%	 < 85ms
	99.9%	 < 110ms
	100.0%	 < 140ms
""";









# for inp in train_dataset:
#     inp.features.detach()

# torch.cuda.empty_cache()

show_audio(multy_context_audio_model, test_dataset, "test")
show_audio(multy_context_audio_model, test_dataset, "test", duration_model=duration_combined_model)

# [DETECTION+DTW]AVERAGE ERROR: 8.37% COUNT:1344
# [DETECTION]AVERAGE ERROR: 8.73% COUNT:1344
# [test] DIFF abs mean: 7.50ms (-0.18) min:0.00ms max:758.12ms
# 	48.9%	 < 5ms		77.4%	 < 10ms
# 	88.5%	 < 15ms		93.0%	 < 20ms
# 	95.4%	 < 25ms		96.8%	 < 30ms
# 	97.7%	 < 35ms		98.3%	 < 40ms
# 	98.7%	 < 45ms		98.9%	 < 50ms
# 	99.1%	 < 55ms		99.2%	 < 60ms
# 	99.3%	 < 65ms		99.4%	 < 70ms
# 	99.4%	 < 75ms		99.5%	 < 80ms
# 	99.5%	 < 85ms		99.6%	 < 90ms
# 	99.6%	 < 95ms		99.7%	 < 100ms
# 	99.7%	 < 105ms		100.0%	 < 9999ms
# 48.89% 77.36% 88.52% 93.05% 95.41% 96.81% 97.71% 98.27% 98.67% 98.89% 99.07% 99.20% 99.31% 99.39% 99.44% 99.48% 99.53% 99.58% 99.62% 99.65%

# [test]
# [duration model]
# danger: dtw_error 31.4% wrong idx:69
# - warped_result: (376, 54)
# - truth:(376, 54)
# danger: dtw_error 31.3% wrong idx:933
# - warped_result: (643, 54)
# - truth:(643, 54)
# [(233.0, 944), (251.75, 371), (340.0, 497), (390.0, 665), (715.0, 933)]
# cache_hits: 0
# [DETECTION+DTW]AVERAGE ERROR: 8.12% COUNT:1344
# [DETECTION]AVERAGE ERROR: 8.73% COUNT:1344
# [test] DIFF abs mean: 6.98ms (0.07) min:0.00ms max:715.00ms
# 	49.3%	 < 5ms		77.9%	 < 10ms
# 	89.0%	 < 15ms		93.6%	 < 20ms
# 	95.9%	 < 25ms		97.3%	 < 30ms
# 	98.1%	 < 35ms		98.7%	 < 40ms
# 	99.0%	 < 45ms		99.2%	 < 50ms
# 	99.4%	 < 55ms		99.5%	 < 60ms
# 	99.6%	 < 65ms		99.7%	 < 70ms
# 	99.7%	 < 75ms		99.8%	 < 80ms
# 	99.8%	 < 85ms		99.8%	 < 90ms
# 	99.8%	 < 95ms		99.9%	 < 100ms
# 	99.9%	 < 105ms		100.0%	 < 9999ms
# 49.30% 77.92% 89.03% 93.57% 95.87% 97.27% 98.11% 98.66% 99.03% 99.24% 99.41% 99.53% 99.60% 99.67% 99.71% 99.75% 99.78% 99.81% 99.83% 99.86%
