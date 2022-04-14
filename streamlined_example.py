from random import choice
from time import sleep
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from load import load_csv, load_files, File, UtteranceBatch, wds_load
from spn.models.base import ModeSwitcherBase
from spn.models.soft_pointer_network import SoftPointerNetwork
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import torch.nn as nn


def fix_borders(generated: List[File], report_error=300):
    output = []
    for item in generated:
        borders = item.output_timestamps.copy()

        switched = False
        prev = 0
        for i, v in enumerate(borders):
            if v < prev:
                switched = True
                after = borders[i + 1] if i + 1 < len(borders) else borders.max()
                v = (prev + after - 0.0001) / 2 if prev < after else prev + 0.001

            borders[i] = v
            prev = v

        item = item.update(output_timestamps=borders)
        output.append(item)

        # TODO: enable?
        # prev = 0
        # for i, v in enumerate(borders):
        #     assert v >= prev, f"This should never happen! {i}"
        #     prev = v

        diff = item.target_timestamps - item.output_timestamps
        if np.abs(diff).max() > report_error:
            print(f"[id:{item.source}]  [{diff.min():5.0f} {diff.max():5.0f}]  {switched}")

    return output


def display_diff(errors, name="", unit="ms", plotting=False):
    errors = errors.copy()
    hist, bins = np.histogram(
        abs(errors),
        # bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 9999],
        bins=[0, 5, 10, 20, 30, 50, 100, 105, 9999],
    )
    hist = np.round(hist / len(errors) * 100, 2)
    hist = np.cumsum(hist)

    print(
        f"[{name}] DIFF abs mean: {abs(errors).mean():.2f}{unit} "
        f"({errors.mean():.2f}) min:{abs(errors).min():.2f}{unit} "
        f"max:{abs(errors).max():.2f}{unit}",
    )
    rows = list(zip(hist, bins, bins[1:]))
    for r in zip(rows[::2], rows[1::2]):
        s = ""
        for h, _b, e in r:
            s += f"\t{h:.2f}%\t < {e:.0f}{unit}\t"
        print(s)

    print(*[f"{h:2.2f}" for h, b, e in rows][:-2], "", sep="% ")
    # print([e for h, b, e in rows])

    if plotting:
        _f, axarr = plt.subplots(1, 2, figsize=(10, 3))
        axarr[0].bar(
            range(len(bins) - 1),
            hist,
        )
        axarr[0].set_xticklabels(bins, fontdict=None, minor=False)
        axarr[1].hist(np.clip(errors, -70, 70), bins=5)


def report_duration(
    generated: List[File], plotting=False
):
    winlen, winstep = generated[0].config
    ms_per_step = winstep * 1000

    # reject last border which is EOF
    diffs = [
        (item.output_durations[:-1] - item.target_durations[:-1])
        * item.target_timestamps[-1]
        * ms_per_step
        for item in generated
    ]

    diff = np.concatenate(diffs)

    print("TOTAL", np.abs(diff).sum(), diff.shape)
    display_diff(diff, "position", plotting=plotting)
    return generated


def report_borders(
    generated: List[File], plotting=False
):
    winlen, winstep = generated[0].config
    ms_per_step = winstep * 1000
    # reject last border which is EOF
    diffs = [item.target_timestamps[:-1] - item.output_timestamps[:-1] for item in generated]
    diff = np.concatenate(diffs) * ms_per_step

    print("TOTAL", np.abs(diff).sum(), diff.shape)
    display_diff(diff, "position", plotting=plotting)
    return generated


class MyCustomDataset(Dataset):
    def __init__(self, files: List[File]):
        self.files = sorted(files, key=lambda x: -len(x.features_spectogram))

    def __getitem__(self, index):
        return self.files[index]

    def __len__(self):
        return len(self.files)

    def features_batch_process(self, batch_features: List[np.array]) -> UtteranceBatch:
        # this is used when a list of data items is transformed into a batch
        # TODO: could we, should we use pack_padded_sequence
        padded = nn.utils.rnn.pad_sequence([torch.tensor(e) for e in batch_features], batch_first=True)
        lens = torch.tensor([len(item) for item in batch_features])
        _b, max_len, *_f = padded.shape
        return UtteranceBatch(
            padded,
            torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1),
            lens,
        )

    def collate_fn(self, batch: List[File]):
        batch = sorted(batch, key=lambda x: -len(x.features_spectogram))
        result = {'original': batch, 'size': len(batch)}
        first = batch[0]
        for k in first.__fields__.keys():
            values = [getattr(item, k) for item in batch]
            result[k] = self.features_batch_process(values) if getattr(values[0], 'dtype', None) in [np.float32,                                                                                                   np.int64] else values
        return result

    def batch(self, batch_size, shuffle=True, num_workers=8, persistent_workers=True):
        return DataLoader(
            dataset=self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn,
            num_workers=num_workers,  persistent_workers=persistent_workers, prefetch_factor=4,
            pin_memory=True,
        )


class MaskedNLL(nn.Module):
    loss = nn.NLLLoss(
        reduction='mean',
        # label_smoothing=0.01,
        ignore_index=-100,
    )

    def forward(self, pred, target, mask, weight=None):
        # pred = torch.mul(pred, mask.unsqueeze(2))
        # print(pred.min(), pred.max())
        pred = torch.log(pred.clip(0.0001, 0.9999))
        target = torch.mul(target, mask) + (~mask) * -100
        loss = self.loss(pred.transpose(1, 2), target)
        return loss


loss_func_nll = MaskedNLL()


def weights_trainer(batch: Dict[str, UtteranceBatch], model: nn.Module, loss_function: nn.Module):
    features_audio = batch['features_spectogram']
    features_transcription = batch['features_phonemes']

    result = model.with_weights(
        features_transcription.padded,
        features_transcription.masks,
        features_audio.padded,
        features_audio.masks,
    )

    target = batch['target_timestamps']

    # reject last border which is EOF
    masks = target.masks.clone()
    # masks.sum() + batch_size should be the original count
    masks[:, :-1] *= masks[:, 1:]
    masks[:, -1] = False

    target_long = target.padded.clone().unsqueeze(-1).round().to(torch.long)

    # target = torch.dstack([target_long]).clip(min=0)
    target = torch.dstack([target_long - 1, target_long, target_long + 1]).clip(min=0)
    # target = torch.dstack([target_long, target_long + 1]).clip(min=0)
    gradient = torch.FloatTensor([-1, 0, 1]).to(result.device)
    # target = torch.dstack([target_long - 2, target_long - 1, target_long, target_long + 1, target_long + 2]).clip(min=0)
    # gradient = torch.FloatTensor([2, 3, 4, 5]).to(result.device)
    # target = torch.dstack([target_long - 2, target_long - 1, target_long, target_long + 1, target_long + 2, target_long + 3]).clip(min=0)
    # gradient = torch.FloatTensor([1, 2, 3, 4, 5, 6]).to(result.device)

    losses = []
    w = torch.take_along_dim(result, target, 2).sum(axis=-1, keepdims=True)
    loss = 1 - torch.mul(w, masks.unsqueeze(-1)).sum() / masks.sum()
    losses.append(loss)
    #
    # delta = batch['target_timestamps'].padded - target_long.squeeze(-1)
    # diff = (torch.take_along_dim(result, target, 2) * gradient).sum(axis=-1) - (delta + 0)
    # loss = torch.pow(diff, 1).abs().sum() / masks.sum()
    # losses.append(loss)

    # .clip(max=.66)
    # loss = loss_func((torch.take_along_dim(result, target, 2) * target).sum(axis=-1), batch['target_timestamps'].padded, masks)
    # losses.append(loss)
    # # diff = (torch.take_along_dim(result, target, 2) * target).sum(axis=-1) - batch['target_timestamps'].padded
    # # diff = torch.mul(diff, masks)
    # # loss = torch.pow(diff, 1).abs().sum() / masks.sum()
    # # losses.append(loss)
    #
    # loss = loss_func_nll(result, target_long.squeeze(-1), masks)
    # losses.append(loss)
    # loss = loss_func_ce(result, target_long.squeeze(-1), masks)
    # losses.append(loss)
    return sum(losses), result, [(result.clone().detach().cpu().numpy(), batch['original'])]


def duration_trainer(batch: Dict[str, UtteranceBatch], model: nn.Module, loss_function: nn.Module):
    features_audio = batch['features_spectogram']
    features_transcription = batch['features_phonemes']
    target = batch['target_durations']

    timestamps = batch['target_timestamps']
    durations = timestamps.padded.max(dim=1).values.unsqueeze(1)

    result = model.with_duration(
        features_transcription.padded,
        features_transcription.masks,
        features_audio.padded,
        features_audio.masks,
    )


    result_local = result.clone().detach().cpu().numpy()
    items = [item.update(output_durations=result_local[i, :end])
             for i, (item, end) in enumerate(zip(batch['original'], target.lengths))]

    return loss_function(result * durations, target.padded * durations, target.masks) / 4, result, items


def argmax_trainer(batch: Dict[str, UtteranceBatch], model: nn.Module, loss_function: nn.Module):
    features_audio = batch['features_spectogram']
    features_transcription = batch['features_phonemes']
    target = batch['target_timestamps']

    result = model.with_argmax(
        features_transcription.padded,
        features_transcription.masks,
        features_audio.padded,
        features_audio.masks,
    )

    result_local = result.clone().detach().cpu().numpy()
    items = [item.update(output_timestamps=result_local[i, :end])
             for i, (item, end) in enumerate(zip(batch['original'], target.lengths))]

    return None, result, items


def position_gradient_trainer(batch: Dict[str, UtteranceBatch], model: nn.Module, loss_function: nn.Module):
    features_audio = batch['features_spectogram']
    features_transcription = batch['features_phonemes']
    target = batch['target_timestamps']

    result = model.with_gradient(
        features_transcription.padded,
        features_transcription.masks,
        features_audio.padded,
        features_audio.masks,
    )

    result_local = result.clone().detach().cpu().numpy()
    items = [item.update(output_timestamps=result_local[i, :end])
             for i, (item, end) in enumerate(zip(batch['original'], target.lengths))]

    # reject last border which is EOF
    masks = target.masks.clone()
    # masks.sum() + batch_size should be the original count
    masks[:, :-1] *= masks[:, 1:]
    masks[:, -1] = False
    weights = None
    # weights = batch['weights_phonemes'].padded
    return loss_function(result, target.padded.clone(), masks, weights=weights), result, items


def occurrence_trainer(batch: Dict[str, UtteranceBatch], model: nn.Module, loss_function: nn.Module):
    features_audio = batch['features_spectogram']
    features_transcription = batch['features_phonemes']
    target = batch['target_occurrences']

    result = model.with_occurrence(
        features_transcription.padded,
        features_transcription.masks,
        features_audio.padded,
        features_audio.masks,
    )

    result_local = result.clone().detach().cpu().numpy()
    items = [item.update(output_occurrences=np.argmax(result_local[i, :end], 1))
             for i, (item, end) in enumerate(zip(batch['original'], target.lengths))]

    return loss_func_ce(result, target.padded, target.masks), result, items


class MaskedMSE(nn.Module):
    mse = nn.MSELoss()

    def forward(self, pred, target, mask, weights=None):
        if weights is not None:
            pred = torch.mul(pred, weights)
            target = torch.mul(target, weights)

        pred = torch.mul(pred, mask)
        target = torch.mul(target, mask)
        return self.mse(pred, target, )


class MaskedCE(nn.Module):
    loss = nn.CrossEntropyLoss(
        reduction='mean',
        # label_smoothing=0.01,
        ignore_index=-100,
    )

    def forward(self, pred, target, mask, weight=None):
        # pred = torch.mul(pred, mask.unsqueeze(2))
        target = torch.mul(target, mask) + (~mask) * -100
        loss = self.loss(pred.transpose(1, 2), target)
        return loss


class DivMaskedMSE(nn.Module):
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    def __init__(self, cutoff, flip=False):
        super().__init__()
        self.cutoff = cutoff
        self.flip = flip
        print("CUTOFF", cutoff)

    def forward(self, pred, target, mask, weights=None):
        # print(weight.shape)
        diff = torch.abs(pred - target)
        if not self.flip:
            diff = diff > self.cutoff
        else:
            diff = diff < self.cutoff

        # pred = pred * weight[:, :-1]
        # target = target * weight[:, :-1]

        mask_diff = mask & diff
        pred = torch.mul(pred, mask_diff)
        target = torch.mul(target, mask_diff)
        mse = self.mse(pred, target)
        # return mse

        mask_diff = mask & ~diff
        pred = torch.mul(pred, mask_diff)
        target = torch.mul(target, mask_diff)
        l1 = self.l1(pred, target)

        return mse + l1


class MaskedL1(nn.Module):
    l1 = nn.L1Loss()

    def forward(self, pred, target, mask, weights=None):
        if weights is not None:
            pred = torch.mul(pred, weights)
            target = torch.mul(target, weights)

        pred = torch.mul(pred, mask)
        target = torch.mul(target, mask)
        return self.l1(pred, target)


loss_func_ce = MaskedCE()
loss_func = MaskedMSE()
# loss_func = MaskedL1()
# loss_func = DivMaskedMSE(1.5, flip=False)


class Thing(ModeSwitcherBase, pl.LightningModule):
    Mode = SoftPointerNetwork.Mode

    def __init__(self):
        super().__init__()
        self.soft_pointer_model = SoftPointerNetwork(54, 26, 32, dropout=0.05)
        self.mode = self.soft_pointer_model.mode

    def training_step(self, batch, batch_idx):
        loss = None
        for method in [
            # choice([
            #     # position_gradient_trainer,
            #     # weights_trainer,
            #     weights_trainer,
            # ]),
            # position_gradient_trainer,
            # occurrence_trainer,
            # duration_trainer,
            weights_trainer,
        ]:
            l, *_ = method(batch, self.soft_pointer_model, loss_func)
            self.log(method.__name__, l, prog_bar=True, batch_size=batch['size'])
            if loss is None:
                loss = l
            else:
                loss += l

        self.log('train_loss', loss, prog_bar=True, batch_size=batch['size'])
        self.log('epoch_loss', loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch['size'])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, *_ = position_gradient_trainer(batch, self.soft_pointer_model, loss_func)
        self.log('val_loss', loss, prog_bar=True, batch_size=batch['size'])
        return loss

    def test_step(self, batch, batch_idx):
        loss, *_ = position_gradient_trainer(batch, self.soft_pointer_model, loss_func)
        self.log('test_loss', loss, prog_bar=True, batch_size=batch['size'])

    def predict_step(self, batch, *_) -> List[File]:
        return self.forward_with(batch)[2]

    def forward_with(self, batch):
        if self.is_argmax:
            return argmax_trainer(batch, self.soft_pointer_model, loss_func)
        if self.is_gradient:
            return position_gradient_trainer(batch, self.soft_pointer_model, loss_func)
        if self.is_duration:
            return duration_trainer(batch, self.soft_pointer_model, loss_func)
        if self.is_occurrence:
            return occurrence_trainer(batch, self.soft_pointer_model, loss_func)
        if self.is_weights:
            return weights_trainer(batch, self.soft_pointer_model, loss_func)

    def forward(self, batch):
        return self.forward_with(batch)[0]

    def configure_optimizers(self):
        # lr = 0.0015
        # lr = 0.0004
        # lr = 0.0001  # down to 17.1 ~ ish
        # lr = 0.00001
        lr = 0.000001  # long time
        # lr = 0.0000001  # oc
        # lr = 0.000000076  # oc
        # lr = 0.00000001
        # lr = 0.000000001
        print("LR", lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.001) # , amsgrad=True)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=5)
        # 0.9 ** (256 / 4 ) = 0.001
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.95)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        # lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=4)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": lr_scheduler,
            #     "monitor": "val_loss",
            # },
        }


# tensorboard --bind_all --logdir lightning_logs/
if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    # class Benchmark(pl.callbacks.Callback):
    #     def __init__(self) -> None:
    #         super().__init__()
    #         self.count = 0
    #
    #     def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #         self.count += 1
    #         lr = trainer.optimizers[0].param_groups[0]['lr']
    #         print("Epoch", self.count, lr)

    trainer = pl.Trainer(
        # gpus=1,
        gpus=2, plugins=pl.plugins.DDPPlugin(find_unused_parameters=True),
        # accumulate_grad_batches=2,
        max_epochs=1200,
        # max_time=dict(seconds=300 * 55),
        # precision=16,  # 23e 8.2s -> 19e 9.2s
        gradient_clip_val=1.5,  # this is very beneficial
        progress_bar_refresh_rate=2,
        log_every_n_steps=8,  # todo: hmm?
        callbacks=[
            # pl.callbacks.StochasticWeightAveraging(swa_epoch_start=1),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.RichProgressBar(),
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints-ce3/", filename="clean-{epoch:02d}-{epoch_loss:.4f}",
                save_top_k=2, monitor="epoch_loss",
            ),
            # pl.callbacks.ModelCheckpoint(
            #     dirpath="checkpoints-ce3/", filename="clean",
            #     save_top_k=1, monitor="epoch_loss",
            # )
        ]
    )

    # module = Thing()
    # module = Thing.load_from_checkpoint("checkpoints-ce3/clean-epoch=533-epoch=epoch_loss=2.1731.ckpt", strict=False)

    # module = Thing.load_from_checkpoint("checkpoints/clean-epoch=48-epoch=epoch_loss=2.9561.ckpt"
    #                                     , strict=False)
    module = Thing.load_from_checkpoint("clean-start-ce3.ckpt", strict=False)
    # module = Thing.load_from_checkpoint("clean-start-mt-p.ckpt", strict=False)
    # module = Thing.load_from_checkpoint("clean-start.ckpt", strict=False)
    # module = Thing.load_from_checkpoint("clean-start-64.ckpt", strict=False)
    # module = Thing.load_from_checkpoint("clean-start-occ.ckpt", strict=False)
    # module = Thing.load_from_checkpoint("clean-start-mt-2.ckpt", strict=False)
    # module = Thing.load_from_checkpoint("example-large-ok-4.ckpt", strict=False)
    # module = Thing.load_from_checkpoint("example-new.ckpt", strict=False)
    # module.soft_pointer_model.load(path="spn/trained_weights/position_model-final.pth")
    # generated = generate_borders(module, test_dataset)
    # generated = fix_borders(generated, report_error=550)
    # report_borders(generated, plotting=False)

    limit = None
    # limit = 256
    # limit = 8

    # test_dataset = MyCustomDataset(wds_load("test_data_0.025wl_0.01ws.tar.xz", limit))
    # train_dataset = MyCustomDataset(wds_load("train_data_0.025wl_0.01ws.tar.xz", limit))

    test_dataset = MyCustomDataset(wds_load("test_data_0.03wl_0.015ws.tar.xz", limit))
    train_dataset = MyCustomDataset(wds_load("train_data_0.03wl_0.015ws.tar.xz", limit))

    # base = ".data"
    # test_files = load_csv(f"{base}/test_data.csv", sa=False)[:limit]
    # test_files = load_files(base, test_files)
    # test_dataset = MyCustomDataset(test_files)
    # # soft_pointer_model.load(path="spn/trained_weights/position_model-final.pth")
    #
    # train_files = load_csv(f"{base}/train_data.csv", sa=False)[:limit]
    # train_files = load_files(base, train_files)
    # train_dataset = MyCustomDataset(train_files)

    # lr 1e-5 bad, 1e-4 good, 1e-3 bad
    # dropout: lower is better?
    # clip val: 3.5 did not affect
    # weights: no real effect?
    # pos embeddings: removed
    # panns

    trainer.fit(module, train_dataset.batch(128))  # , test_dataset.batch(128, shuffle=False))
    sleep(2)
    trainer.save_checkpoint(f"clean-start-ce3.ckpt")

    generated = sum(trainer.predict(module.with_gradient, dataloaders=test_dataset.batch(128, shuffle=False)),
                    start=[])
    generated = fix_borders(generated, report_error=550)
    report_borders(generated, plotting=False)

    # generated = sum(trainer.predict(module.with_duration, dataloaders=test_dataset.batch(128, shuffle=False)),
    #                 start=[])
    # report_duration(generated)
    #
    generated = sum(trainer.predict(module.with_gradient, dataloaders=train_dataset.batch(128, shuffle=False)),
                    start=[])
    generated = fix_borders(generated, report_error=550)
    report_borders(generated, plotting=False)
