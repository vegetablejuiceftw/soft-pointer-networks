from typing import Dict

import torch
from torch import nn
import pytorch_lightning as pl

from dataloading import dto
from spnz.base import ModeSwitcherBase, ExportImportMixin
from spnz.components import Encoder
from spnz.panns import SpectogramCNN

import logging


logging.basicConfig(level=logging.INFO)

class MaskedNLL(nn.Module):
    loss = nn.NLLLoss(
        reduction='mean',
        ignore_index=-100,
    )

    def forward(self, pred, target, mask, weight=None):
        pred = torch.log(pred.clip(0.0001, 0.9999))
        target = torch.mul(target, mask) + (~mask) * -100
        loss = self.loss(pred.transpose(1, 2), target)
        return loss


class MaskedCE(nn.Module):
    loss = nn.CrossEntropyLoss(
        reduction='mean',
        label_smoothing=0.01,
        ignore_index=-100,
    )

    def forward(self, pred, target, mask, weight=None):
        target = torch.mul(target, mask) + (~mask) * -100
        loss = self.loss(pred.transpose(1, 2), target)
        return loss


class MaskedMSE(nn.Module):
    mse = nn.MSELoss()

    def forward(self, pred, target, mask, weights=None):
        if weights is not None:
            pred = torch.mul(pred, weights)
            target = torch.mul(target, weights)

        pred = torch.mul(pred, mask)
        target = torch.mul(target, mask)
        return self.mse(pred, target, )


loss_func_nll = MaskedNLL()
# loss_func_ce = MaskedCE()
loss_func_mse = MaskedMSE()


def att_forward(output, mask_output, context, mask_context):
    # https://arxiv.org/abs/1706.03762
    # context & mask is what we attend to
    batch_size, hidden_size, input_size = output.size(0), output.size(2), context.size(1)
    # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
    # matrix by matrix product https://pytorch.org/docs/stable/torch.html#torch.bmm
    attn = torch.bmm(output, context.transpose(1, 2))
    # TODO: scale step missing?

    if mask_context is not None:
        if mask_output is not None:
            attn = attn.transpose(1, 2)
            attn.data.masked_fill_(~mask_output.unsqueeze(1), 0)
            attn = attn.transpose(1, 2)

        attn.data.masked_fill_(~mask_context.unsqueeze(1), -float('inf'))

    attn = torch.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
    return attn


def att_forward(output, mask_output, context, mask_context):
    # https://arxiv.org/abs/1706.03762
    # context & mask is what we attend to
    batch_size, hidden_size, input_size = output.size(0), output.size(2), context.size(1)
    # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
    # matrix by matrix product https://pytorch.org/docs/stable/torch.html#torch.bmm
    attn = torch.bmm(output, context.transpose(1, 2))
    # TODO: scale step missing?

    if mask_context is not None:
        if mask_output is not None:
            attn = attn.transpose(1, 2)
            attn.data.masked_fill_(~mask_output.unsqueeze(1), 0)
            attn = attn.transpose(1, 2)

        attn.data.masked_fill_(~mask_context.unsqueeze(1), -float('inf'))

    attn = torch.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
    return attn


def simple_att(model, encoded_phonemes, masks_phonemes, encoded_audio, mask_audio):
    activation = nn.Tanh()  # 13, 21
    encoded_phonemes = activation(encoded_phonemes)
    encoded_audio = activation(encoded_audio)

    w_slice = att_forward(encoded_phonemes, masks_phonemes, encoded_audio, mask_audio)
    return w_slice


def progressive_masking_att(model, encoded_phonemes, masks_phonemes, encoded_audio, mask_audio):
    activation = nn.Tanh()  # 13, 21
    # activation = nn.Hardtanh() # 17
    # activation = nn.GELU()  # 20
    # activation = nn.SiLU() #
    # activation = nn.Sigmoid() # 19
    # activation = nn.Hardswish() # 16

    encoded_phonemes = activation(encoded_phonemes)
    encoded_audio = activation(encoded_audio)

    # tensor to store decoder outputs
    batch_size, out_seq_len, _ = encoded_phonemes.shape
    w = torch.zeros(batch_size, out_seq_len, encoded_audio.shape[1], device=encoded_audio.device)
    w_mask_history, w_mask, iter_mask_audio = [], None, mask_audio

    for t in range(out_seq_len):
        iter_input = encoded_phonemes[:, t:(t + 1), :]

        if len(w_mask_history) == model.history_size:
            w_mask = w_mask_history[0]
            iter_mask_audio = mask_audio * (w_mask > model.min_activation) if mask_audio is not None else w_mask > model.min_activation

        iter_mask_transcription = masks_phonemes[:, t:(t + 1)] if masks_phonemes is not None else None

        # _, w_slice = model.multihead_attn(
        #     iter_input,
        #     encoded_audio,
        #     encoded_audio,  # index gradient?
        # )

        w_slice = att_forward(iter_input, iter_mask_transcription, encoded_audio, iter_mask_audio)

        w[:, t:(t + 1), :] = w_slice

        w_slice_mask = torch.cumsum(w_slice.squeeze(1).clone(), dim=1).detach()
        if model.history_size:
            w_mask_history = w_mask_history[-model.history_size + 1:] + [w_slice_mask]

    return w


def predict_weights(batch: dto.FileBatch, model: 'Thing'):
    ms_per_step = model.panns.ms_per_step
    activation = nn.GELU()  # 7, 9, 10

    audio = batch.audio.padded
    result = model.panns(audio, interpolate=False)

    encoded_audio = result['spectogram']
    # encoded_audio = activation(model.encoder_audio.handle(activation(encoded_audio))[0])  # + activation(encoded_audio)
    encoded_audio = model.spectogram_mixer(torch.cat([
        activation(model.encoder_audio.handle(activation(encoded_audio))[0]),
        activation(encoded_audio),
    ], dim=2))
    encoded_audio = model.panns.interpolate(encoded_audio)

    mask_audio = torch.ones(size=encoded_audio.shape[:2], dtype=torch.bool, device=encoded_audio.device)
    for i, item in enumerate(batch.original):
        samples = item.audio.shape[-1]
        steps = int(samples / model.panns.sample_rate * 1000 / ms_per_step)
        mask_audio[i, steps + 1:] = False
    encoded_audio = torch.mul(encoded_audio, mask_audio.unsqueeze(2))

    ids_phonemes = batch.phonetic_detail.id
    features_phonemes = model.encoder_transcription_embedding(ids_phonemes.padded)
    # features_phonemes[:, 1:] += features_phonemes[:, :-1]
    # features_phonemes[:, :-1] += features_phonemes[:, 1:]
    masks_phonemes = ids_phonemes.mask.clone()
    # masks.sum() + batch_size should be the original count
    masks_phonemes[:, :-1] *= masks_phonemes[:, 1:].clone()
    masks_phonemes[:, -1] = False
    encoded_phonemes, _ = model.encoder_transcription(batch.phonetic_detail.id.update(padded=features_phonemes))
    # encoded_phonemes = features_phonemes
    # encoded_phonemes = activation(encoded_phonemes)

    logging.debug([encoded_audio.shape, encoded_phonemes.shape])

    if model.history_size:
        attention = progressive_masking_att(model, encoded_phonemes, masks_phonemes, encoded_audio, mask_audio)
    else:
        attention = simple_att(model, encoded_phonemes, masks_phonemes, encoded_audio, mask_audio)

    attention = attention * masks_phonemes.unsqueeze(2) * mask_audio.unsqueeze(1)

    mask_window = None

    target = batch.phonetic_detail.stop

    losses = []

    loss_names = [
        "gradient_full",
        "interpolation_weight",
        # "gradient_context",
        # "nll",
        # "nll", "nll_noisy",
    ]

    if "gradient_full" in loss_names:
        # full gradient->timestamp loss
        gradient = torch.ones_like(attention).cumsum(dim=-1) - 1
        predicted = (attention * gradient * ms_per_step).sum(dim=-1)
        diff: torch.Tensor = (predicted - target.padded) / ms_per_step
        # diff = torch.log1p((predicted - target.padded).abs())
        # diff = (torch.log1p(predicted) - torch.log1p(target.padded)).abs()
        diff = diff.abs() * masks_phonemes
        diff = diff.clip(min=0, max=1.0)
        # TODO: CLIP LOSS to 1.0?

        # diff = diff ** 2
        loss = diff.sum() / masks_phonemes.sum()
        losses.append(loss)

    if "nll" in loss_names:
        steps = target.padded.clone() / ms_per_step
        if "nll_noisy" in loss_names:
            noise = torch.randn_like(steps) / 8
            dx = .51
            # noise = noise.clip(min=-dx, max=dx)
            noise[noise > dx] = 0
            noise[noise < -dx] = 0
            steps += noise
        target_long = steps.unsqueeze(-1).round().to(torch.long).clip(min=0, max=attention.shape[-1] - 1)
        losses.append(loss_func_nll(attention, target_long.squeeze(-1), masks_phonemes))

    if "gradient_context" in loss_names:
        ####""" target_long = steps.round().unsqueeze(-1).to(torch.long)"""
        target_long = (target.padded / ms_per_step).clone().unsqueeze(-1).to(torch.long)
        # target_index = torch.dstack([target_long - 1, target_long, target_long + 1]).clip(min=0, max=attention.shape[-1] - 1)
        target_index = torch.dstack([target_long - 2, target_long - 1, target_long, target_long + 1, target_long + 2]).clip(min=0, max=attention.shape[-1] - 1)
        # target_index = torch.dstack([target_long, target_long + 1]).clip(min=0, max=attention.shape[-1] - 1)
        predicted = (torch.take_along_dim(attention, target_index, 2) * target_index).sum(axis=-1) * ms_per_step

        diff = (predicted - target.padded) / ms_per_step
        # diff = torch.log1p(diff)
        # diff = (torch.log1p(predicted) - torch.log1p(target.padded)).abs()
        # diff = diff.abs()
        # diff = diff ** 2
        diff = diff.abs() * masks_phonemes
        diff = diff.clip(min=0, max=1.0)
        loss = diff.sum() / masks_phonemes.sum()  # is mask handled here nicely?
        losses.append(loss)

    if "interpolation_weight" in loss_names:
        target_steps = (target.padded / ms_per_step).clone()
        # noise = torch.randn_like(target_steps) / 8
        # dx = .04
        # noise[noise > dx] = 0
        # noise[noise < -dx] = 0
        # target_steps += noise
        target_floor = target_steps.floor()
        target_delta = target_steps - target_floor
        target_long = target_floor.unsqueeze(-1).to(torch.long).clip(min=0, max=attention.shape[-1] - 1).clone()

        target_index_l = torch.dstack([target_long]).clip(min=0, max=attention.shape[-1] - 1)
        target_index_r = torch.dstack([target_long + 1]).clip(min=0, max=attention.shape[-1] - 1)
        #
        # target_index_l = torch.dstack([target_long - 1, target_long]).clip(min=0, max=attention.shape[-1] - 1)
        # target_index_r = torch.dstack([target_long + 1, target_long + 2]).clip(min=0, max=attention.shape[-1] - 1)

        predicted_l = (torch.take_along_dim(attention, target_index_l, 2)).sum(axis=-1)
        predicted_r = (torch.take_along_dim(attention, target_index_r, 2)).sum(axis=-1)

        diff = (predicted_r - target_delta).abs() + (predicted_l + target_delta - 1).abs()
        diff = diff * masks_phonemes
        # diff = diff ** 2
        loss = diff.sum() / masks_phonemes.sum()
        losses.append(loss)

    return dict(
        loss=sum(losses),
        attention=attention,
        batch=batch,
        window=mask_window,
    )


def predict_gradient(batch: dto.FileBatch, model: 'Thing'):
    attention: torch.FloatTensor = predict_weights(batch, model)['attention']

    ms_per_step = model.panns.ms_per_step
    position_gradient = (torch.ones_like(attention[0, 0, :]).cumsum(0) - 1) * ms_per_step
    timestamps = (position_gradient.unsqueeze(1) * attention.transpose(1, 2)).sum(1).detach()

    predictions = [
        file.update(output_timestamps=timestamps[i, :len(file.phonetic_detail.id)])
        for i, file in enumerate(batch.original)
    ]
    # dict(
    #     timestamps=timestamps,
    #     attention=attention,
    #     batch=batch,
    #     predictions=predictions,
    # )
    return batch.update(
        attention=attention,
        original=predictions,
        output_timestamps=timestamps,
    )


class Thing(ExportImportMixin, ModeSwitcherBase, pl.LightningModule):
    class Mode(ModeSwitcherBase.Mode):
        weights = "weights"
        gradient = "gradient"
        occurrence = "occurrence"
        argmax = "argmax"
        argmax_gradient = "argmax_gradient"
        duration = "duration"

    def __init__(self, dropout=0.3):
        super().__init__()
        self.mode = self.Mode.weights
        self.min_activation = 0.1
        self.history_size = 2
        # self.history_size = None
        print("history_size", self.history_size)

        hidden_size = 128
        mel_bins = 64
        self.panns = SpectogramCNN(
            sample_rate=16000, window_size=1024, hop_size=256,
            mel_bins=mel_bins, fmin=0, fmax=14000,
            classes_num=hidden_size,
            interpolate_ratio=1,  # to upscale the CNN down sampling
            dropout=dropout,
        )
        # self.panns.att_block.activation = "linear"

        print("ms_per_step", self.panns.ms_per_step)
        self.spectogram_mixer = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.encoder_audio = Encoder(
            hidden_size=hidden_size,
            embedding_size=hidden_size,
            out_dim=hidden_size,
            num_layers=2,
            dropout=dropout,
        )

        # embedding_size = 32
        embedding_size = hidden_size
        self.encoder_transcription_embedding = torch.nn.Embedding(40, embedding_size, max_norm=True, padding_idx=-1)

        self.encoder_transcription = Encoder(
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            out_dim=hidden_size,
            num_layers=2,
            dropout=dropout,
        )

        # self.multihead_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True, dropout=dropout)

    def training_step(self, batch, batch_idx):
        loss = []
        batch_size = len(batch.source)
        for method in [
            predict_weights,
        ]:
            method_loss = method(batch, self)['loss']
            self.log(method.__name__, method_loss, prog_bar=True, batch_size=batch_size)
            loss.append(method_loss)
        loss = sum(loss)
        self.log('train_loss', loss, prog_bar=True, batch_size=batch_size)
        self.log('epoch_loss', loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        return loss

    def forward_with(self, batch: dto.FileBatch):
        if self.is_weights:
            return predict_weights(batch, self)
        if self.is_argmax:
            return predict_argmax(batch, self)
        if self.is_gradient:
            return predict_gradient(batch, self)
        if self.is_argmax_gradient:
            return predict_argmax_gradient(batch, self)

    def forward(self, batch):
        return self.forward_with(batch)

    def configure_optimizers(self):
        # lr = 0.0015
        # lr = 0.0009
        # lr = 0.0004  # can overfit 4.3ms
        # lr = 0.0001  # down to 17.1 ~ ish
        # lr = 0.00004
        lr = 0.00001  # NlLL ok
        # lr = 0.000001  # too low?
        print("LR", lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.4, amsgrad=True)
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.1)
        return {
            "optimizer": optimizer,
        }


if __name__ == '__main__':
    import os

    os.chdir(os.getcwd().split('/spnz')[0])
    module = Thing()
    model_data = torch.load('.data2/Cnn14_DecisionLevelAtt_mAP=0.425.pth', map_location=torch.device('cpu'))['model']
    # model_data = torch.load('panns.pth', map_location=torch.device('cpu'))#['model']
    # # # print(model_data.keys())
    # for key in tuple(model_data):
    #     # if any(k in key for k in "fc1 att_block".split()):
    #     if any(k in key for k in "spectrogram_extractor.stft.conv_ logmel_extractor.melW".split()):
    #         del model_data[key]
    module.panns.load_state_dict(model_data, strict=False)
    print(module)
"""
history_size None
ms_per_step 64.0
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
1500it [00:00, 1645.88it/s]
100%|██████████| 1184/1184 [00:00<00:00, 14727.90it/s]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
Predicting DataLoader 0: 100%|██████████| 10/10 [00:08<00:00,  1.15it/s]
1184
TOTAL 544344.25 (43197,)
[position] DIFF abs mean: 12.60ms (0.39) min:0.00ms max:261.71ms
	23.55%	 < 5ms		46.56%	 < 10ms
	84.03%	 < 20ms		95.82%	 < 30ms
	98.77%	 < 50ms		99.81%	 < 100ms
	99.82%	 < 105ms		100.00%	 < 9999ms
1680it [00:01, 1636.30it/s]
100%|██████████| 1336/1336 [00:00<00:00, 15748.65it/s]
Predicting: 0it [00:00, ?it/s]LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
Predicting DataLoader 0: 100%|██████████| 11/11 [00:10<00:00,  1.05it/s]
1336
TOTAL 1016184.56 (48857,)
[position] DIFF abs mean: 20.80ms (0.68) min:0.00ms max:905.33ms
	20.54%	 < 5ms		40.83%	 < 10ms
	73.28%	 < 20ms		86.61%	 < 30ms
	93.17%	 < 50ms		97.02%	 < 100ms
	97.23%	 < 105ms		99.98%	 < 9999ms

Process finished with exit code 0

[position] DIFF abs mean: 1.85ms (-0.06) min:0.00ms max:77.20ms
	93.56%	 < 5ms		98.88%	 < 10ms
	99.86%	 < 20ms		99.96%	 < 30ms
	99.99%	 < 50ms		99.99%	 < 100ms
	99.99%	 < 105ms		99.99%	 < 9999ms
[position] DIFF abs mean: 14.62ms (1.22) min:0.00ms max:1020.31ms
	38.67%	 < 5ms		63.70%	 < 10ms
	83.89%	 < 20ms		90.82%	 < 30ms
	95.61%	 < 50ms		98.24%	 < 100ms
	98.33%	 < 105ms		99.99%	 < 9999ms
"""
