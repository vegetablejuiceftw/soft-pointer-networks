import pytorch_lightning as pl

from dataloading import dto
from dataloading.dataset import UtteranceDataset
from spnz.model import Thing

from typing import List, Dict

import numpy

import numpy as np
import matplotlib.pyplot as plt


def detach(tensor) -> numpy.ndarray:
    if hasattr(tensor, 'detach'): tensor = tensor.detach()
    if hasattr(tensor, 'cpu'): tensor = tensor.cpu()
    if hasattr(tensor, 'numpy'): tensor = tensor.numpy()
    tensor = tensor.copy()
    return tensor


def fix_borders(generated: List[dto.File], report_error=300):
    output = []
    for item in generated:
        borders = detach(item.output_timestamps)
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

        # diff = item.phonetic_detail.stop - item.output_timestamps
        # if np.abs(diff).max() > report_error:
        #     print(f"[id:{item.source}]  [{diff.min():5.0f} {diff.max():5.0f}]  {switched}")

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

    # print(*[f"{h:2.2f}" for h, b, e in rows][:-2], "", sep="% ")
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
    generated: List[dto.File], plotting=False
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
    generated: List[dto.File], plotting=False
):
    # reject last border which is EOF
    diffs = [item.phonetic_detail.stop[:-1] - item.output_timestamps[:-1] for item in generated]
    diff = np.concatenate(diffs)

    print("TOTAL", np.abs(diff).sum(), diff.shape)
    display_diff(diff, "position", plotting=plotting)
    return generated


if __name__ == '__main__':
    import warnings
    import os

    os.chdir(os.getcwd().split('/spnz')[0])
    warnings.filterwarnings("ignore")

    # limit = 1500
    limit = 8

    # module = Thing.load_from_checkpoint("clean-start-ce3.ckpt", strict=False)
    module = Thing()

    trainer = pl.Trainer(accelerator='gpu', devices=[1])

    for split in "test", "train":
        file_path = f".data/{split}_data.tar.xz"
        dataset = UtteranceDataset(dto.wds_load(file_path, limit=limit))

        for batch in dataset.batch(batch_size=8, mp=False):
            print(batch)
            print(batch.audio.padded.shape)
            print(batch.phonetic_detail.stop.padded.shape)
            break

        # 42ms / 20ms
        result = trainer.predict(module.with_gradient, dataloaders=dataset.batch(4, shuffle=False))
        generated = sum([r.original for r in result], start=[])
        print(len(generated))
        generated = fix_borders(generated, report_error=550)
        report_borders(generated, plotting=False)

        # # 20ms / 10ms
        # generated = sum(trainer.predict(module.with_duration, dataloaders=dataset.batch(128, shuffle=False)), start=[])
        # report_duration(generated)
        #
        # # 0.70 / 0.78
        # generated = sum(trainer.predict(module.with_occurrence, dataloaders=dataset.batch(128, shuffle=False)), start=[])
        # diffs = [item.target_occurrences == item.output_occurrences for item in generated]
        # diff = np.concatenate(diffs)
        # print(diff.mean().round(3))
        # np.set_printoptions(edgeitems=30, linewidth=100000)
        # # for file in generated[:8]:
        # #     print()
        # #     print(file.target_occurrences)
        # #     print(file.output_occurrences)
