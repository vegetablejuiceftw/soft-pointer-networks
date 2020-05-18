import torch
import torch.nn as nn
import torch.nn.functional as F

from components import Attention, Encoder, PositionalEncoding
from models.base import ExportImportMixin, ModeSwitcherBase


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
        self.use_iter = True
        self.use_pos_encode = True
        self.use_pre_transformer = True

        self.t_transformer = nn.Sequential(
            nn.Linear(embedding_transcription_size, 32),
            nn.Sigmoid(),
            nn.Linear(32, embedding_transcription_size),
            nn.Sigmoid()
        ).to(device)

        self.a_transformer = nn.Sequential(
            nn.Linear(embedding_audio_size, 32),
            nn.Sigmoid(),
            nn.Linear(32, embedding_audio_size),
            nn.Sigmoid()
        ).to(device)

        self.encoder_transcription = Encoder(
            hidden_size=hidden_size,
            embedding_size=embedding_transcription_size,
            out_dim=hidden_size,
            num_layers=2,
            dropout=dropout,
            time_scale=time_transcription_scale)

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
        batch, trans_len, seq_len = weights.shape

        if position_encodings:
            position_encoding = self.pos_encode(torch.zeros(batch, seq_len, self.position_encoding_size))
            positions = torch.bmm(weights, position_encoding)
            return positions[:, :-1]

        if argmax:
            return weights.max(2)[1][:, :-1]

        positions = (self.gradient[:seq_len] * weights.transpose(1, 2)).sum(1)[:, :-1]
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
        encoder_audio_outputs, _ = self.encoder_audio(
            features_audio,
            skip_pos_encode=not self.use_pos_encode
        )

        if not self.use_iter:
            # not progressive batching
            w = self.attn(
                F.tanh(encoder_transcription_outputs), mask_transcription,
                F.tanh(encoder_audio_outputs), mask_audio)

        else:
            encoder_transcription_outputs = F.relu(encoder_transcription_outputs)
            encoder_audio_outputs = F.relu(encoder_audio_outputs)
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


if __name__ == '__main__':
    print(SoftPointerNetwork(54, 26, 256, device='cpu'))
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
