from random import randint
from typing import Dict

import torch
from torch import nn
import pytorch_lightning as pl

from load import load_csv, load_files, wds_load, UtteranceBatch
from spn.models.base import ModeSwitcherBase, ExportImportMixin
from spn.models.panns import Cnn14_DecisionLevelAtt
from streamlined_example import MyCustomDataset

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


def predict_weights(batch: Dict[str, UtteranceBatch], model: 'Thing'):
    ms_per_step = model.panns.ms_per_step
    activation = nn.GELU()

    audio = batch['features_audio'].padded
    result = model.panns(audio, interpolate=False)

    encoded_audio = result['framewise_output']
    # encoded_audio = activation(model.encoder_audio.handle(activation(encoded_audio))[0]) #+ encoded_audio
    encoded_audio = model.spectogram_mixer(torch.cat([
        activation(model.encoder_audio.handle(activation(encoded_audio))[0]),
        activation(encoded_audio),
    ], dim=2))
    encoded_audio = model.panns.interpolate(encoded_audio)

    # spectogram = result['spectogram']
    # logging.debug("spectogram %s", spectogram.shape)
    # encoded_audio = model.spectogram_mixer(torch.cat([encoded_audio, spectogram], dim=2))

    # encoded_audio = model.encoder_audio.handle(encoded_audio)[0]

    mask_audio = torch.ones(size=encoded_audio.shape[:2], dtype=torch.bool, device=encoded_audio.device)
    for i, item in enumerate(batch['original']):
        samples = item.features_audio.shape[-1]
        steps = int(samples / model.panns.sample_rate * 1000 / ms_per_step)
        mask_audio[i, steps + 1:] = False
    encoded_audio = torch.mul(encoded_audio, mask_audio.unsqueeze(2))

    features_phonemes = batch['features_phonemes']
    masks_phonemes = features_phonemes.masks.clone()
    # masks.sum() + batch_size should be the original count
    masks_phonemes[:, :-1] *= masks_phonemes[:, 1:]
    masks_phonemes[:, -1] = False
    encoded_phonemes, _ = model.encoder_transcription(features_phonemes)

    logging.debug([encoded_audio.shape, encoded_phonemes.shape])

    attn_output, attention = model.multihead_attn(
        activation(encoded_phonemes),
        activation(encoded_audio),
        encoded_audio,
    )

    attention = attention * masks_phonemes.unsqueeze(2) * mask_audio.unsqueeze(1)

    mask_window = None

    # if model.training:
    #     attention = attention.clip(max=0.90)

    # if randint(0, 3) == 9 or not model.training:
    #     window = 4
    #     cumsum = (1 - torch.cumsum(attention, dim=2)) * masks_phonemes.unsqueeze(2)
    #     mask_window = cumsum + 0.01
    #     for i in range(2, window):
    #         mask_window += cumsum.roll(-i, 1)
    #     mask_window = mask_window > 2
    #     mask_window[:, -window:] = 1
    #     attention *= mask_window * masks_phonemes.unsqueeze(2)
    #
    #     # window = 5
    #     # mask_window = 1 - torch.cumsum(attention, dim=2).roll(-window, 1)#.detach()
    #     # mask_window[:, -window:] = 1
    #     # attention = attention * mask_window
    #
    #     if not model.training:
    #         attention = attention / attention.sum(dim=2).detach().unsqueeze(2).clip(min=0.00000000000001)

    # attention = attention * masks_phonemes.unsqueeze(2) * mask_audio.unsqueeze(1)

    logging.debug("OUT: %s attention:%s", attn_output.shape, attention.shape)

    target = batch['target_timestamps']

    steps = target.padded.clone() / ms_per_step
    steps += (torch.randn_like(steps) / 8 ).clip(min=-0.5, max=0.5)

    losses = []
    # gradient = torch.ones_like(attention).cumsum(dim=-1) - 1
    # predicted = (attention * gradient * ms_per_step).sum(dim=-1)
    # diff = (predicted - target.padded) / 512
    # diff = torch.pow(diff, 2)
    # loss = diff.sum() / target.masks.sum()
    # losses.append(loss)

    # target_long = steps.clone().unsqueeze(-1).to(torch.long)
    # losses.append(loss_func_nll(attention, target_long.squeeze(-1), target.masks))
    ### main loos func
    target_long = steps.unsqueeze(-1).round().to(torch.long).clip(min=0, max=attention.shape[-1] - 1)
    losses.append(loss_func_nll(attention, target_long.squeeze(-1), masks_phonemes))

    # target_long = steps.clone().unsqueeze(-1).to(torch.long)
    # target = torch.dstack([target_long, target_long + 1]).clip(min=0)
    # w = torch.take_along_dim(attention, target, 2).sum(axis=-1, keepdims=True)
    # # loss = 1 - w.sum()
    # # loss = 1 - torch.mul(w, masks.unsqueeze(-1)).sum() / masks.sum()
    # loss = (1 - torch.mul(w, masks.unsqueeze(-1))).sum()
    # losses.append(loss)

    # weighted_timestamp = (torch.take_along_dim(attention, target, 2) * target).sum(axis=-1)
    # loss = loss_func_mse(weighted_timestamp, batch['target_timestamps'].padded, masks)
    # losses.append(loss)

    # target_long = steps.round().unsqueeze(-1).to(torch.long)
    # # # # target_index = torch.dstack([target_long])
    # target_index = torch.dstack([target_long - 1, target_long, target_long + 1]).clip(min=0, max=attention.shape[-1] - 1)
    # # target_index = torch.dstack([target_long - 2, target_long - 1, target_long, target_long + 1, target_long + 2]).clip(min=0, max=attention.shape[-1] - 1)
    # # #
    # # # #### ALT loss func
    # # target_long = target.padded.clone().unsqueeze(-1).to(torch.long)
    # # target_index = torch.dstack([target_long, target_long + 1]).clip(min=0, max=attention.shape[-1] - 1)
    # predicted = (torch.take_along_dim(attention, target_index, 2) * target_index).sum(axis=-1) * ms_per_step
    # diff = (predicted - target.padded)
    # diff = torch.pow(diff, 2)
    # # diff = diff.abs()
    # loss = diff.sum() / masks_phonemes.sum()
    # losses.append(loss)

    return dict(
        loss=sum(losses),
        attention=attention,
        batch=batch,
        window=mask_window,
    )


def predict_argmax(batch: Dict[str, UtteranceBatch], model: 'Thing'):
    attention: torch.FloatTensor = predict_weights(batch, model)['attention']

    ms_per_step = model.panns.ms_per_step
    argmaxed = attention.argmax(2) * ms_per_step

    predictions = [
        file.update(output_timestamps=argmaxed[i, :len(file.features_phonemes)])
        for i, file in enumerate(batch['original'])
    ]

    return dict(
        timestamps=argmaxed,
        attention=attention,
        batch=batch,
        predictions=predictions,
    )


def predict_gradient(batch: Dict[str, UtteranceBatch], model: 'Thing'):
    attention: torch.FloatTensor = predict_weights(batch, model)['attention']

    ms_per_step = model.panns.ms_per_step
    position_gradient = (torch.ones_like(attention[0, 0, :]).cumsum(0) - 1) * ms_per_step
    timestamps = (position_gradient.unsqueeze(1) * attention.transpose(1, 2)).sum(1).detach()

    predictions = [
        file.update(output_timestamps=timestamps[i, :len(file.features_phonemes)])
        for i, file in enumerate(batch['original'])
    ]

    return dict(
        timestamps=timestamps,
        attention=attention,
        batch=batch,
        predictions=predictions,
    )


def predict_argmax_gradient(batch: Dict[str, UtteranceBatch], model: 'Thing'):
    attention: torch.FloatTensor = predict_weights(batch, model)['attention']

    ms_per_step = model.panns.ms_per_step
    argmaxed = attention.max(2)

    target_long = argmaxed.indices
    # scope of the interpolation
    target_index = torch.dstack([target_long - 1, target_long, target_long + 1]).clip(min=0, max=attention.shape[-1] - 1)
    # target_index = torch.dstack([target_long - 2, target_long - 1, target_long, target_long + 1, target_long + 2]).clip(min=0)
    target_weights = torch.take_along_dim(attention, target_index, 2)
    # normalize the probabilities in the scope
    target_weights = target_weights / target_weights.sum(axis=-1).unsqueeze(-1).clip(min=0.001)
    predicted = (target_weights * target_index).sum(axis=-1) * ms_per_step

    predictions = [
        file.update(output_timestamps=predicted[i, :len(file.features_phonemes)])
        for i, file in enumerate(batch['original'])
    ]

    return dict(
        attention=attention,
        batch=batch,
        predictions=predictions,
    )


class Encoder(nn.Module):

    def __init__(
        self,
        hidden_size,
        embedding_size,
        out_dim=None,
        num_layers=2,
        dropout=0.4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batchnorm = nn.BatchNorm1d(embedding_size)
        # Embedding layer that will be shared with Decoder
        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * num_layers, out_dim or hidden_size)

    def handle(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.batchnorm(x)
        x = x.permute(0, 2, 1).contiguous()
        x, hidden = self.gru(x)
        x = self.fc(x)
        return x, hidden

    def forward(self, batch: UtteranceBatch):
        x, mask = batch.padded, batch.masks
        x, hidden = self.handle(x)
        x = torch.mul(x, mask.unsqueeze(-1))
        return x, hidden


class Thing(ExportImportMixin, ModeSwitcherBase, pl.LightningModule):
    class Mode(ModeSwitcherBase.Mode):
        weights = "weights"
        gradient = "gradient"
        occurrence = "occurrence"
        argmax = "argmax"
        argmax_gradient = "argmax_gradient"
        duration = "duration"

    def __init__(self, dropout=0.40):
        super().__init__()
        self.mode = self.Mode.weights

        hidden_size = 256
        mel_bins = 64
        self.panns = Cnn14_DecisionLevelAtt(
            sample_rate=16000, window_size=1024,
            hop_size=96, mel_bins=mel_bins, fmin=50, fmax=14000,
            classes_num=hidden_size,
            interpolate_ratio=2,  # to upscale the CNN down sampling
            dropout=dropout,
        )
        print("ms_per_step", self.panns.ms_per_step)
        self.spectogram_mixer = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

        self.encoder_audio = Encoder(
            hidden_size=hidden_size,
            embedding_size=hidden_size,
            out_dim=hidden_size,
            num_layers=2,
            dropout=dropout,
        )

        self.encoder_transcription = Encoder(
            hidden_size=128,
            embedding_size=54,
            out_dim=hidden_size,
            num_layers=2,
            dropout=dropout,
        )
        self.panns.att_block.activation = "linear"

        self.multihead_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True, dropout=dropout)

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

    def forward_with(self, batch):
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
        # lr = 0.025
        # lr = 0.0015
        # lr = 0.0009
        # lr = 0.0004  # can overfit 4.3ms
        lr = 0.0001  # down to 17.1 ~ ish
        # lr = 0.00001  # NLL ok
        # lr = 0.000001
        # lr = 0.00000001
        print("LR", lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.001, amsgrad=True)
        return {
            "optimizer": optimizer,
        }


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    trainer = pl.Trainer(
        # gpus=1,
        gpus=2, plugins=pl.plugins.DDPPlugin(find_unused_parameters=True),
        max_epochs=100,
        # precision=16,  # 23e 8.2s -> 19e 9.2s
        gradient_clip_val=1.5,  # this is very beneficial
        progress_bar_refresh_rate=1,
        log_every_n_steps=8,  # todo: hmm?
        callbacks=[
            # pl.callbacks.StochasticWeightAveraging(swa_epoch_start=1),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            # pl.callbacks.RichProgressBar(),
        ]
    )

    limit, split = None, "train"
    # limit, split = 128, "train"

    file_path = f"{split}_data.tar.xz"
    files = wds_load(file_path, limit)
    dataset = MyCustomDataset(files)

    module = Thing.load_from_checkpoint("panns.ckpt", strict=False)
    # module = Thing().load('thing.pth', 'fc1'.split())

    # module = Thing()
    # model_data = torch.load('Cnn14_DecisionLevelAtt_mAP=0.425.pth', map_location=torch.device('cpu'))['model']
    # # print(model_data.keys())
    # for key in tuple(model_data):
    #     if 'att_block' in key:
    #         del model_data[key]
    # module.panns.load_state_dict(model_data, strict=False)

    trainer.fit(module, dataset.batch(16))  # , test_dataset.batch(128, shuffle=False))
    import time
    time.sleep(1)
    print(trainer.save_checkpoint(f"panns.ckpt"))
