from datetime import timedelta
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
        bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 9999],
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
    return loss_function(result, target.padded, masks), result


class MaskedMSE(nn.Module):
    mse = nn.MSELoss()

    def forward(self, pred, target, mask, *_):
        pred = torch.mul(pred, mask)
        target = torch.mul(target, mask)
        return self.mse(pred, target)


import pytorch_lightning as pl

loss_func = MaskedMSE()


class Thing(pl.LightningModule):

    def __init__(self):
        super().__init__()
        soft_pointer_model = SoftPointerNetwork(54, 26, 256, dropout=0.05)
        soft_pointer_model.load(path="spn/trained_weights/position_model-final.pth")
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25)
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
    file_path = f"{base}/test_data.csv"
    test_files = load_csv(file_path, sa=False)[:limit]
    test_files = load_files(base, test_files)
    test_dataset = MyCustomDataset(test_files)

    module = Thing()
    # generated = generate_borders(module, test_dataset, batch_size=128)
    # generated = fix_borders(generated, report_error=550)
    # report_borders(generated, plotting=False)

    # lr 1e-5 bad, 1e-4 good, 1e-3 bad
    # features_transcription +next
    trainer = pl.Trainer(
        gpus=1,
        # gpus=2, plugins=pl.plugins.DDPPlugin(find_unused_parameters=False),
        max_epochs=26,
        max_time=timedelta(seconds=480),
        # precision=16,  # 23e 8.2s -> 19e 9.2s
        gradient_clip_val=0.5,  # this is very beneficial
        progress_bar_refresh_rate=2,
        log_every_n_steps=8,  # todo: hmm?
        callbacks=[
            pl.callbacks.StochasticWeightAveraging(swa_lrs=0.00001),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.RichProgressBar(),
        ]
    )
    trainer.fit(module, test_dataset.batch(64), test_dataset.batch(128, shuffle=False))

    generated = generate_borders(module, test_dataset)
    generated = fix_borders(generated, report_error=550)
    report_borders(generated, plotting=False)
