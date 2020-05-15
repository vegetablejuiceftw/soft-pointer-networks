from dependencies import *


class PositionSimple(nn.Module):
    class Mode:
        weights = "weights"
        position = "position"
        gradient = "gradient"
        argmax = "argmax"

    def __getattr__(self, item):
        cleaned = item.replace("is_", "").replace("with_", "")
        if hasattr(self.Mode, cleaned):
            mode = getattr(self.Mode, cleaned)
            if "is_" in item:
                return self.mode == mode
            elif "with_" in item:
                self.mode = mode
                return self
        return super().__getattr__(item)

    def __dir__(self):
        return super().__dir__() + [f"is_{k}" for k in self.Mode.keys] + [f"with_{k}" for k in self.Mode.keys]

    def __init__(self, embedding_size, embedding_audio_size, hidden_size, vocab_size, device, attention_size=None,
                 dropout=0.35):
        super().__init__()
        out_dim = hidden_size  # vocab_size
        self.encoder_transcription = Encoder(hidden_size, embedding_size, out_dim=out_dim, num_layers=2,
                                             dropout=dropout, time_scale=POS_TRANSCRIPTION_SCALE)

        self.encoder_audio = Encoder(hidden_size, embedding_audio_size, out_dim=out_dim, num_layers=2, dropout=dropout,
                                     time_scale=POS_SCALE)

        # self.encoder_transcription_2 = Encoder(hidden_size, out_dim, out_dim=vocab_size, num_layers=2, dropout=dropout, time_scale=None)
        # self.encoder_audio_2 = Encoder(hidden_size, out_dim, out_dim=vocab_size, num_layers=2, dropout=dropout, time_scale=POS_SCALE)

        self.attn = Attention(None)
        self.gradient = (torch.cumsum(torch.ones(2 ** 14), 0).unsqueeze(1) - 1).cuda()
        self.zero = torch.zeros(256, 2048, vocab_size).to(device)
        self.pos_encode = PositionalEncoding(vocab_size, dropout, scale=POS_SCALE)

        print("scale:", POS_TRANSCRIPTION_SCALE)
        self.vocab_size = vocab_size
        self.device = device
        self.to(device)
        self.mode = self.Mode.gradient
        self.flags = {}
        self.use_iter = True
        self.use_pos_encode = True

        self.t_transformer = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.Sigmoid(),
            nn.Linear(32, embedding_size),
            nn.Sigmoid()
        ).to(device)

        self.a_transformer = nn.Sequential(
            nn.Linear(embedding_audio_size, 32),
            nn.Sigmoid(),
            nn.Linear(32, embedding_audio_size),
            nn.Sigmoid()
        ).to(device)

    def weights_to_positions(self, weights, argmax=False):
        batch_size, audio_size, input_size = weights.shape

        batch, trans_len, seq_len = weights.shape
        if argmax:
            return weights.max(2)[1][:, :-1]
        positions = (self.gradient[:seq_len] * weights.transpose(1, 2)).sum(1)[:, :-1]
        return positions

    def forward(self, features_transcription, mask_transcription, features_audio, mask_audio):
        features_transcription = features_transcription.clone()
        features_transcription[:, :-1] += features_transcription[:, 1:] * 0.55

        features_transcription = self.t_transformer(features_transcription)
        features_audio = self.a_transformer(features_audio)

        encoder_transcription_outputs, _ = self.encoder_transcription(features_transcription,
                                                                      skip_pos_encode=not self.use_pos_encode)  # # # #

        encoder_audio_outputs, _ = self.encoder_audio(features_audio, skip_pos_encode=not self.use_pos_encode)

        if not self.use_iter:
            # not progressive batching
            w = self.attn(F.tanh(encoder_transcription_outputs), mask_transcription, F.tanh(encoder_audio_outputs),
                          mask_audio)
        else:
            encoder_transcription_outputs = F.relu(encoder_transcription_outputs)
            encoder_audio_outputs = F.relu(encoder_audio_outputs)
            # tensor to store decoder outputs
            batch_size, out_seq_len, _ = features_transcription.shape
            w = torch.zeros(batch_size, out_seq_len, features_audio.shape[1]).to(self.device)

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

                    # iter_memory = iter_memory * (w_mask.unsqueeze(2) * (1 - pad) + pad)
                # print(iter_input.shape, mask_transcription.shape, (iter_memory).shape, iter_mask_audio.shape)
                iter_mask_transcription = mask_transcription[:, t:(t + 1)] if mask_transcription is not None else None
                w_slice = self.attn(iter_input, iter_mask_transcription, (iter_memory), iter_mask_audio)

                if w_mask is not None:
                    w[:, t:(t + 1), :] = w_slice * w_mask.unsqueeze(1)
                else:
                    w[:, t:(t + 1), :] = w_slice

                w_mask = w_slice.squeeze(1).clone()
                w_mask = torch.cumsum(w_mask, dim=1).detach()
                w_masks.append(w_mask)
                w_masks = w_masks[-2:]

        if self.is_weights:
            return w

        if self.is_gradient or self.is_argmax:
            return self.weights_to_positions(w, argmax=self.is_argmax)

        batch, seq_len, dimensions = encoder_audio_outputs.shape
        processsed_audio = self.zero[:batch, :seq_len, :self.vocab_size]
        pos = self.pos_encode(processsed_audio)
        position_encodes = torch.bmm(w, pos)

        if self.is_position:
            return position_encodes[:, :-1]


if __name__ == '__main__':
    pass
    """
    position_model = PositionSimple(KNOWN_LABELS_COUNT, INPUT_SIZE, 256, POS_DIM, device).to(device)
    # train(position_model, 10, toy_dataset.batch(64), toy_dataset.batch(128), loss_function=MaskedMSE(), train_function=position_gradient_trainer, lr_decay=0.91, lr=0.00151)
    # load(position_model, "/content/drive/My Drive/dataset/position_model-reforged-repeat-10.pth") # 7 is ok, 8, 10 is cool,
    # load(position_model, "/content/drive/My Drive/dataset/position_model-normalized-2.pth") # 7 is ok, 8, 10 is cool,  1.5
    # load(position_model, "/content/drive/My Drive/dataset/position_model-pure-6.pth")
    # load(position_model, "/content/drive/My Drive/dataset/position_model-pure-derp.pth")
    # load(position_model, "/content/drive/My Drive/dataset/position_model-pure-12.pth")
    # load(position_model, "/content/drive/My Drive/dataset/position_model-revive-2.pth")
    # load(position_model, "/content/drive/My Drive/dataset/position_model-simple-final.pth")
    # load(position_model, "/content/drive/My Drive/dataset/position_model-final.pth")
    # load(position_model, "/content/drive/My Drive/dataset/position_model-final-2.pth")
    # load(position_model, "/content/drive/My Drive/dataset/position_model-final-3.pth")
    load(position_model, "/content/drive/My Drive/dataset/position_model-final-4.pth")

    # load(position_model, "/content/drive/My Drive/dataset/position_model-derp.pth")

    """
    load(position_model, "/content/drive/My Drive/dataset/position_model-final-4.pth")
    TOTAL
    344903.16
    [position]
    DIFF
    abs
    mean: 7.01
    ms(-0.39)
    min: 0.00
    ms
    max: 780.91
    ms
    55.97 % < 5
    ms
    81.93 % < 10
    ms
    90.88 % < 15
    ms
    94.61 % < 20
    ms
    96.62 % < 25
    ms
    97.79 % < 30
    ms
    98.48 % < 35
    ms
    98.87 % < 40
    ms
    99.15 % < 45
    ms
    99.35 % < 50
    ms
    99.48 % < 55
    ms
    99.57 % < 60
    ms
    99.65 % < 65
    ms
    99.67 % < 70
    ms
    99.70 % < 75
    ms
    99.72 % < 80
    ms
    99.74 % < 85
    ms
    99.76 % < 90
    ms
    99.77 % < 95
    ms
    99.79 % < 100
    ms
    99.81 % < 105
    ms
    100.01 % < 9999
    ms
55.97 % 81.93 % 90.88 % 94.61 % 96.62 % 97.79 % 98.48 % 98.87 % 99.15 % 99.35 % 99.48 % 99.57 % 99.65 % 99.67 % 99.70 % 99.72 % 99.74 % 99.76 % 99.77 % 99.79 %
"""
position_model.with_gradient


def eval_border_agreement(duration_combined_model=None):
    show_position_batched(position_model.with_gradient, test_dataset,
                          duration_combined_model=duration_combined_model)
    show_position_batched(position_model.with_gradient, train_dataset,
                          duration_combined_model=duration_combined_model)
    if duration_combined_model is not None:
        print(" -  -  -  -  -  - WITHOUT -  -  -  -  -  - ")
        show_position_batched(position_model.with_gradient, test_dataset)
        show_position_batched(position_model.with_gradient, train_dataset)


# evaluate(position_model, train_dataset.batch(64), loss_function=MaskedMSE(), train_function=position_gradient_trainer) # 0.15
# evaluate(position_model, test_dataset.batch(64), loss_function=MaskedMSE(), train_function=position_gradient_trainer) # 2.1
eval_border_agreement(duration_combined_model=duration_combined_model)
torch.cuda.empty_cache()
!nvidia - smi

torch.cuda.empty_cache()

# toy_dataset = DirectMaskDataset(train_files, limit=2043)
work_dataset = train_dataset
# work_dataset = train_augment_dataset

evaluation = [toy_dataset.batch(64)]
evaluation = [test_dataset.batch(64), train_eval_dataset.batch(64)]

train_batch_size = 32
if WIN_STEP < 0.010:
    train_batch_size = 16

f = .35
n = 1
# eval_border_agreement(duration_combined_model=duration_combined_model)

# train(position_model.with_gradient, int(10 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=MaskedMSE(), train_function=position_gradient_trainer, lr_decay=0.951, lr= f * 0.000161 * 0.131 * .10, weight_decay=1e-05 * 14)
# eval_border_agreement(duration_combined_model=duration_combined_model)

# train(position_model.with_position, int(8 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=CosineLoss(), train_function=position_encode_trainer, lr_decay=0.93, lr= f * 0.000051 * 0.005)
# eval_border_agreement(duration_combined_model=duration_combined_model)

# train(position_model.with_gradient, int(8 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=MaskedMSE(), train_function=position_gradient_trainer, lr_decay=0.991, lr= f * 0.000161 * 0.131 * 3.1 * 0.01, weight_decay=1e-05 * 14)

# export_model(position_model, "/content/drive/My Drive/dataset/position_model-pure-derp.pth")
# show_position_batched(position_model, toy_dataset, report_error=750)
# eval_border_agreement(duration_combined_model=duration_combined_model)

# NORMAL
# train(position_model.with_gradient, int(10 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=MaskedMSE(), train_function=position_gradient_trainer, lr_decay=0.98, lr= f * 0.000161 * 0.131 * .01, weight_decay=1e-05 * 14)
# eval_border_agreement(duration_combined_model=duration_combined_model)

# train(position_model.with_gradient, int(9 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=DivMaskedMSE(10, flip=True), train_function=position_gradient_trainer, lr_decay=0.98, lr= f * 0.000161 * 0.131 * .05, weight_decay=1e-05 * 14)
# eval_border_agreement(duration_combined_model=duration_combined_model)

# train(position_model.with_gradient, int(30 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=DivMaskedMSE(13, flip=False), train_function=position_gradient_trainer, lr_decay=0.98, lr= f * 0.000161 * 0.131 * .2, weight_decay=1e-05 * 14)
# eval_border_agreement(duration_combined_model=duration_combined_model)

# train(position_model.with_gradient, int(30 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=DivMaskedMSE(5, flip=False), train_function=position_gradient_trainer, lr_decay=0.98, lr= f * 0.000161 * 0.131 * .2, weight_decay=1e-05 * 14)
# eval_border_agreement(duration_combined_model=duration_combined_model)

# train(position_model.with_gradient, int(30 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=DivMaskedMSE(21, flip=False), train_function=position_gradient_trainer, lr_decay=0.98, lr= f * 0.000161 * 0.131 * .2, weight_decay=1e-05 * 14)
# eval_border_agreement(duration_combined_model=duration_combined_model)

# torch.cuda.empty_cache()
# for i in range(10, 0, -1):
#     f = .25
#     # n= 0.8
#     load(position_model, "/content/drive/My Drive/dataset/position_model-final-3.pth")
#     train(position_model.with_gradient, int(5 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=DivMaskedMSE(i, flip=False), train_function=position_gradient_trainer, lr_decay=0.985, lr= f * 0.000161 * 0.131 * .2, weight_decay=1e-05 * 14)
#     eval_border_agreement(duration_combined_model=duration_combined_model)

torch.cuda.empty_cache()
f = .023
n = 1.8
# load(position_model, "/content/drive/My Drive/dataset/position_model-final-3.pth")
train(position_model.with_gradient, int(5 * n), work_dataset.batch(train_batch_size), evaluation,
      loss_function=DivMaskedMSE(1.5, flip=False), train_function=position_gradient_trainer, lr_decay=0.985,
      lr=f * 0.000161 * 0.131 * .2, weight_decay=1e-05 * 14)
# train(position_model.with_gradient, int(5 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=MaskedMSE(), train_function=position_gradient_trainer, lr_decay=0.985, lr= f * 0.000161 * 0.131 * .2, weight_decay=1e-05 * 14)
eval_border_agreement(duration_combined_model=None)
# export_model(position_model, "/content/drive/My Drive/dataset/position_model-derp.pth")

# SIMPLE
# train(position_model.with_gradient, int(30 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=MaskedMSE(), train_function=position_gradient_trainer, lr_decay=0.98, lr= f * 0.000161 * 0.131 * 3.1, weight_decay=1e-05 * 14)
# eval_border_agreement(duration_combined_model=duration_combined_model)
# train(position_model.with_gradient, int(10 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=DivMaskedMSE(5), train_function=position_gradient_trainer, lr_decay=0.98, lr= f * 0.000161 * 0.131 * .1, weight_decay=1e-05 * 14)
# eval_border_agreement(duration_combined_model=duration_combined_model)
# train(position_model.with_gradient, int(10 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=DivMaskedMSE(5), train_function=position_gradient_trainer, lr_decay=0.98, lr= f * 0.000161 * 0.131 * .1, weight_decay=1e-05 * 14)
# train(position_model.with_position, int(30 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=CosineLoss(), train_function=position_encode_trainer, lr_decay=0.93, lr= f * 0.000051 * 0.03)
# eval_border_agreement(duration_combined_model=duration_combined_model)

torch.cuda.empty_cache()

# 49.97% 77.86% 89.11% 93.87% 96.27% 97.61% 98.35% 98.84% 99.13% 99.33% 99.46% 99.57% 99.61% 99.69% 99.73% 99.75% 99.77% 99.78% 99.79% 99.81%
# 56.05% 81.85% 90.93% 94.65% 96.66% 97.77% 98.47% 98.88% 99.16% 99.34% 99.49% 99.54% 99.62% 99.65% 99.68% 99.70% 99.72% 99.74% 99.75% 99.77%
# % % % % % % % % % % % % % % % % % % % %
# 55.97% 81.93% 90.88% 94.61% 96.62% 97.79% 98.48% 98.87% 99.15% 99.35% 99.48% 99.57% 99.65% 99.67% 99.70% 99.72% 99.74% 99.76% 99.77% 99.79%

train(position_model.with_gradient, int(5 * n), work_dataset.batch(train_batch_size), evaluation,
      loss_function=DivMaskedMSE(1.5, flip=False), train_function=position_gradient_trainer, lr_decay=0.985,
      lr=f * 0.000161 * 0.131 * .2, weight_decay=1e-05 * 14)
# train(position_model.with_gradient, int(5 * n), work_dataset.batch(train_batch_size), evaluation, loss_function=MaskedMSE(), train_function=position_gradient_trainer, lr_decay=0.985, lr= f * 0.000161 * 0.131 * .2, weight_decay=1e-05 * 14)
eval_border_agreement(duration_combined_model=None)

work_dataset = train_augment_dataset
train(position_model.with_gradient, int(10 * n), work_dataset.batch(train_batch_size), evaluation,
      loss_function=DivMaskedMSE(5, flip=False), train_function=position_gradient_trainer, lr_decay=0.985,
      lr=f * 0.000161 * 0.131 * .1, weight_decay=1e-05 * 14)
eval_border_agreement(duration_combined_model=duration_combined_model)

"""
