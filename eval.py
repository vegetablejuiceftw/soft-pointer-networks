import numpy as np
from load import load_csv, load_files, wds_load, File
from streamlined_example import MyCustomDataset, Thing, fix_borders, report_borders, report_duration
import pytorch_lightning as pl


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    limit = 1500
    # limit = 8

    module = Thing.load_from_checkpoint("checkpoints-ce/clean-epoch=194-epoch=epoch_loss=0.3884.ckpt", strict=False)

    trainer = pl.Trainer(accelerator='gpu', devices=[1])

    for split in "test", "train":
        # base = ".data"
        # files = load_csv(f"{base}/{split}_data.csv", sa=False)[:limit]
        # files = load_files(base, files)
        # file_path = f"{split}_data_0.025wl_0.01ws.tar.xz"
        file_path = f"{split}_data_0.03wl_0.015ws.tar.xz"
        files = wds_load(file_path, limit)
        dataset = MyCustomDataset(files)
        # 42ms / 20ms
        generated = sum(trainer.predict(module.with_gradient, dataloaders=dataset.batch(128, shuffle=False)), start=[])
        # generated = sum(trainer.predict(module.with_argmax, dataloaders=dataset.batch(128, shuffle=False)), start=[])
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
