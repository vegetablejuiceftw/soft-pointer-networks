from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from load import load_csv, load_files, File, UtteranceBatch
from spn.models.soft_pointer_network import SoftPointerNetwork
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
import torch.nn as nn


def fix_borders(
    generated: List[File], report_error=300
):
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


def generate_borders(
    model, dataset, batch_size: int = 32
):
    model.eval()
    output = []
    with torch.no_grad():
        for batch in tqdm(dataset.batch(batch_size)):
            borders_predicted = model(batch).cpu().detach().numpy()

            for i, item in enumerate(batch['original']):
                item: File
                length = batch['target_timestamps'].lengths[i]
                predicted_border = borders_predicted[i, :length]
                output.append(item.update(output_timestamps=predicted_border))

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
        self.files = files

    def __getitem__(self, index):
        return self.files[index]

    def __len__(self):
        return len(self.files)

    def features_batch_process(self, batch_features: List[np.array]) -> UtteranceBatch:
        # this is used when a list of data items is transformed into a batch
        # TODO: could we, should we use pack_padded_sequence
        padded = nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
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
            result[k] = self.features_batch_process(values) if torch.is_tensor(values[0]) else values
        return result

    def batch(self, batch_size, shuffle=True, num_workers=8,  persistent_workers=True):
        return DataLoader(
            dataset=self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn,
            num_workers=num_workers,  persistent_workers=persistent_workers,
        )


def position_gradient_trainer(batch: Dict[str, UtteranceBatch], model: nn.Module, loss_function: nn.Module):
    features_audio = batch['features_spectogram']
    features_transcription = batch['features_phonemes']
    target = batch['target_timestamps']
    # weights = batch['weights_phonemes']

    result = model(
        features_transcription.padded,
        features_transcription.masks,
        features_audio.padded,
        features_audio.masks,
    )
    # reject last border which is EOF
    masks = target.masks
    # masks.sum() + batch_size should be the original count
    masks[:, :-1] *= masks[:, 1:]
    masks[:, -1] = False
    # , weights = weights.padded
    return loss_function(result, target.padded, masks), result


class MaskedMSE(nn.Module):
    mse = nn.MSELoss()

    def forward(self, pred, target, mask, weights=None):
        if weights is not None:
            pred = torch.mul(pred, weights)
            target = torch.mul(target, weights)
            # pred = pred * weights
            # target = target * weights

        pred = torch.mul(pred, mask)
        target = torch.mul(target, mask)
        return self.mse(pred, target)


class DivMaskedMSE(nn.Module):
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    def __init__(self, cutoff, flip=False):
        super().__init__()
        self.cutoff = cutoff
        self.flip = flip
        print("CUTOFF", cutoff)

    def forward(self, pred, target, mask, *weight):
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

    def forward(self, pred, target, mask, *_):
        pred = torch.mul(pred, mask)
        target = torch.mul(target, mask)
        return self.l1(pred, target)


import pytorch_lightning as pl

loss_func = MaskedMSE()
# loss_func = MaskedL1()
# loss_func = DivMaskedMSE(1.5, flip=False)


class Thing(pl.LightningModule):

    def __init__(self):
        super().__init__()
        soft_pointer_model = SoftPointerNetwork(54, 26, 256, dropout=0.05)
        # soft_pointer_model.load(path="spn/trained_weights/position_model-final.pth")
        self.soft_pointer_model = soft_pointer_model

    def training_step(self, batch, batch_idx):
        loss, _ = position_gradient_trainer(batch, self.soft_pointer_model.with_gradient, loss_func)
        self.log('train_loss', loss, batch_size=batch['size'])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = position_gradient_trainer(batch, self.soft_pointer_model.with_gradient, loss_func)
        self.log('val_loss', loss, batch_size=batch['size'])
        return loss

    def test_step(self, batch, batch_idx):
        loss, _ = position_gradient_trainer(batch, self.soft_pointer_model.with_gradient, loss_func)
        self.log('test_loss', loss, batch_size=batch['size'])

    def forward(self, batch):
        _, result = position_gradient_trainer(batch, self.soft_pointer_model.with_gradient, loss_func)
        return result

    def configure_optimizers(self):
        # lr = 0.00001
        lr = 0.000001  # long time
        # lr = 0.0000001
        print("LR", lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            },
        }


# tensorboard --logdir lightning_logs/
if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    limit = None
    # limit = 16

    base = ".data"
    test_files = load_csv(f"{base}/test_data.csv", sa=False)[:limit]
    test_files = load_files(base, test_files)
    test_dataset = MyCustomDataset(test_files)

    module = Thing.load_from_checkpoint("example3.ckpt")
    generated = generate_borders(module, test_dataset)
    generated = fix_borders(generated, report_error=550)
    report_borders(generated, plotting=False)

    train_files = load_csv(f"{base}/train_data.csv", sa=False)[:limit]
    train_files = load_files(base, train_files)
    train_dataset = MyCustomDataset(train_files)

    # lr 1e-5 bad, 1e-4 good, 1e-3 bad
    # dropout: lower is better?
    # clip val: 3.5 did not affect
    # weights: no real effect?
    # pos embeddings: removed
    # panns
    class Benchmark(pl.callbacks.Callback):

        def __init__(self, dataset) -> None:
            super().__init__()
            self.dataset = dataset
            self.count = 0

        def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            self.count += 1
            # generated = generate_borders(pl_module, self.dataset, batch_size=256)
            # generated = fix_borders(generated, report_error=550)
            print("Epoch", self.count)
            # report_borders(generated, plotting=False)

    trainer = pl.Trainer(
        gpus=1,
        # gpus=2, plugins=pl.plugins.DDPPlugin(find_unused_parameters=False),
        max_epochs=10,
        # max_time=dict(seconds=300 * 1),
        # precision=16,  # 23e 8.2s -> 19e 9.2s
        gradient_clip_val=0.5,  # this is very beneficial
        progress_bar_refresh_rate=2,
        log_every_n_steps=8,  # todo: hmm?
        callbacks=[
            # pl.callbacks.StochasticWeightAveraging(swa_epoch_start=1),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.RichProgressBar(),
            # Benchmark(test_dataset),
        ]
    )
    trainer.fit(module, train_dataset.batch(64), test_dataset.batch(128, shuffle=False))
    trainer.save_checkpoint("example3.ckpt")

    generated = generate_borders(module, test_dataset)
    generated = fix_borders(generated, report_error=550)
    report_borders(generated, plotting=False)
