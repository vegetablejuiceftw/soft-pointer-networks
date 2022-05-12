from load import wds_load
from panns_example import MyCustomDataset, Thing
import pytorch_lightning as pl
import warnings
from streamlined_example import fix_borders, display_diff
from typing import List
from load import File
import numpy as np

warnings.filterwarnings("ignore")


def report_borders(
    generated: List[File], plotting=False
):
    # reject last border which is EOF
    diffs = [item.target_timestamps[:-1] - item.output_timestamps[:-1] for item in generated]
    # diffs = [item.target_timestamps - item.output_timestamps for item in generated]
    diff = np.concatenate(diffs)

    print("TOTAL", np.abs(diff).sum(), diff.shape)
    display_diff(diff, "position", plotting=plotting)
    return generated


module = Thing.load_from_checkpoint("panns.ckpt", strict=False)
# module = Thing().load('thing.pth', 'fc1'.split())

trainer = pl.Trainer(accelerator='gpu', devices=[1])
# trainer = pl.Trainer(accelerator='cpu')

# limit, split = 1500, "test"
# limit, split = 1000, "train"
# limit, split = 32, "train"
# limit, split = 16, "test"
for split in "test", "train":
    limit = 128
    # limit = 512
    # limit = 1512
    file_path = f"{split}_data.tar.xz"
    files = wds_load(file_path, limit)
    dataset = MyCustomDataset(files)

    generated = [file for batch in trainer.predict(module.with_argmax_gradient, dataloaders=dataset.batch(32, shuffle=False, mp=False)) for
                 file in batch['predictions']]
    generated = fix_borders(generated, report_error=2600)
    report_borders(generated, plotting=False)

    # 42ms / 20ms
    # generated = [file for batch in trainer.predict(module.with_gradient, dataloaders=dataset.batch(32, shuffle=False, mp=False)) for
    #              file in batch['predictions']]
    # generated = fix_borders(generated, report_error=1500)
    # report_borders(generated, plotting=False)
    #
    # # # 42ms / 20ms
    # generated = [file for batch in trainer.predict(module.with_argmax, dataloaders=dataset.batch(64, shuffle=False, mp=False)) for
    #              file in batch['predictions']]
    # generated = fix_borders(generated, report_error=2600)
    # report_borders(generated, plotting=False)

