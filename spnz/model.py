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


def progressive_masking_att(model, encoded_phonemes, masks_phonemes, encoded_audio, mask_audio):
    activation = nn.Tanh()

    encoded_phonemes = activation(encoded_phonemes)
    encoded_audio = activation(encoded_audio)

    # tensor to store decoder outputs
    batch_size, out_seq_len, _ = encoded_phonemes.shape
    w = torch.zeros(batch_size, out_seq_len, encoded_audio.shape[1]).to(encoded_audio.device)
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
        w_mask_history = w_mask_history[-model.history_size + 1:] + [w_slice_mask]

    return w


def predict_weights(batch: dto.FileBatch, model: 'Thing'):
    ms_per_step = model.panns.ms_per_step
    activation = nn.GELU()

    audio = batch.audio.padded
    result = model.panns(audio, interpolate=False)

    # encoded_audio = result['framewise_output']
    encoded_audio = result['spectogram']
    # encoded_audio = activation(model.encoder_audio.handle(activation(encoded_audio))[0]) #+ encoded_audio
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
    masks_phonemes = ids_phonemes.mask.clone()
    # masks.sum() + batch_size should be the original count
    masks_phonemes[:, :-1] *= masks_phonemes[:, 1:]
    masks_phonemes[:, -1] = False
    encoded_phonemes, _ = model.encoder_transcription(batch.phonetic_detail.id.update(padded=features_phonemes))

    logging.debug([encoded_audio.shape, encoded_phonemes.shape])

    attention = progressive_masking_att(model, encoded_phonemes, masks_phonemes, encoded_audio, mask_audio)

    attention = attention * masks_phonemes.unsqueeze(2) * mask_audio.unsqueeze(1)

    mask_window = None

    target = batch.phonetic_detail.stop

    losses = []

    gradient = torch.ones_like(attention).cumsum(dim=-1) - 1
    predicted = (attention * gradient * ms_per_step).sum(dim=-1)
    diff = (predicted - target.padded).abs() / 16
    # diff = torch.log1p((predicted - target.padded).abs())
    # diff = (torch.log1p(predicted) - torch.log1p(target.padded)).abs()
    loss = diff.sum() / target.mask.sum()
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

    def __init__(self, dropout=0.1):
        super().__init__()
        self.mode = self.Mode.weights
        self.min_activation = 0.1
        self.history_size = 3

        hidden_size = 128
        mel_bins = 64
        self.panns = SpectogramCNN(
            sample_rate=16000, window_size=1024, hop_size=128,
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

        self.encoder_transcription_embedding = torch.nn.Embedding(40, 32, max_norm=True, padding_idx=-1)

        self.encoder_transcription = Encoder(
            hidden_size=128,
            embedding_size=32,
            out_dim=hidden_size,
            num_layers=2,
            dropout=dropout,
        )

        # self.multihead_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True, dropout=dropout)

    def training_step(self, batch, batch_idx):
        loss = []
        for method in [
            predict_weights,
        ]:
            method_loss = method(batch, self)['loss']
            self.log(method.__name__, method_loss, prog_bar=True, batch_size=batch['size'])
            loss.append(method_loss)
        loss = sum(loss)
        self.log('train_loss', loss, prog_bar=True, batch_size=batch['size'])
        self.log('epoch_loss', loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch['size'])
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
        lr = 0.0004  # can overfit 4.3ms
        # lr = 0.0001  # down to 17.1 ~ ish
        # lr = 0.00001  # NlLL ok
        print("LR", lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01, amsgrad=True)
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
