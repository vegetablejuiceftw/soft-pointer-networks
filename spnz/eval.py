import pytorch_lightning as pl

from dataloading import dto
from dataloading.dataset import UtteranceDataset
from dataloading.utils import detach
from dataloading import transforms
from spnz.model import Thing

from typing import List, Dict

import numpy as np


def fix_borders(generated: List[dto.File]):
    output = []
    for item in generated:
        borders = detach(item.output_timestamps)
        prev = 0
        for i, v in enumerate(borders):
            if v < prev:
                after = borders[i + 1] if i + 1 < len(borders) else borders.max()
                v = (prev + after - 0.0001) / 2 if prev < after else prev + 0.001

            borders[i] = v
            prev = v

        item = item.update(output_timestamps=borders)
        output.append(item)

    return output


def display_diff(errors, name="", unit="ms", plotting=False):
    errors = errors.copy()
    hist, bins = np.histogram(
        abs(errors),
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
    diffs = [item.phonetic_detail.stop[:-1] - detach(item.output_timestamps[:-1]) for item in generated]
    diff = np.concatenate(diffs)

    print("TOTAL", np.abs(diff).sum(), diff.shape)
    display_diff(diff, "position", plotting=plotting)
    return generated


if __name__ == '__main__':
    import warnings
    import os

    os.chdir(os.getcwd().split('/spnz')[0])
    warnings.filterwarnings("ignore")

    limit = 1500
    limit = 512
    # limit = 32
    # limit = None

    # module = Thing.load_from_checkpoint("clean-start-ce3.ckpt", strict=False)
    # module = Thing()
    module = Thing.load_from_checkpoint("panns128.ckpt", strict=False)
    # module = Thing.load_from_checkpoint("panns128-1,55ms-14,9ms.ckpt", strict=False)
    # module = Thing.load_from_checkpoint("panns-1000.ckpt", strict=False)
    # module = Thing.load_from_checkpoint("lightning_logs/version_3/checkpoints/epoch=4-step=320.ckpt", strict=False)
    # module.min_activation = 0.1
    # module.history_size = 2
    # module.history_size = None

    module.eval()

    trainer = pl.Trainer(accelerator='gpu', devices=[1])

    for split in "train", "test":
        file_path = ".data/%s_data.tar.xz"
        dataset = dto.wds_load(file_path % split, limit=limit)
        dataset = dto.apply(dataset, transforms.DefaultTransform().handle)

        # dataset = [d for d in dataset if "SA" not in d.source]
        # dataset = dto.apply(dataset, pad_audio)
        dataset = UtteranceDataset(dataset)

        # 42ms / 20ms
        result = trainer.predict(module.with_gradient, dataloaders=dataset.batch(128, shuffle=False))
        generated = sum([r.original for r in result], start=[])
        print(len(generated))
        # generated = fix_borders(generated, report_error=550)
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
