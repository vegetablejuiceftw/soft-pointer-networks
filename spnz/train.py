import pytorch_lightning as pl
import torch
import time
import numpy as np

from dataloading import dto
from dataloading.dataset import UtteranceDataset
from spnz.eval import fix_borders, report_borders
from spnz.model import Thing


if __name__ == '__main__':
    import warnings
    import os

    os.chdir(os.getcwd().split('/spnz')[0])
    warnings.filterwarnings("ignore")

    trainer = pl.Trainer(
        devices=2, accelerator="gpu",
        # gpus=1,
        # gpus=2, strategy="ddp",
        # gpus=2, plugins=pl.strategies.DDPStrategy(),  # (remove when no torch no grad)
        # gpus=2, plugins=pl.plugins.DDPPlugin(find_unused_parameters=False),  # (remove when no torch no grad)
        # max_epochs=40 * 8,
        max_epochs=8,
        # precision=16,  # 23e 8.2s -> 19e 9.2s
        gradient_clip_val=1.5,  # this is very beneficial
        # progress_bar_refresh_rate=1,
        log_every_n_steps=8,  # todo: hmm?
        callbacks=[
            # pl.callbacks.StochasticWeightAveraging(swa_epoch_start=1),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            # pl.callbacks.RichProgressBar(),
        ]
    )

    limit = None
    # limit = 1500
    # limit = 2
    # limit = 32

    # checkpoint = "panns-1000.ckpt"
    checkpoint = "panns128.ckpt"
    # module = Thing()
    module = Thing.load_from_checkpoint("panns128.ckpt", strict=False)
    # module.history_size = None

    # model_data = torch.load('.data2/Cnn14_DecisionLevelAtt_mAP=0.425.pth', map_location=torch.device('cpu'))['model']
    # module.panns.load_state_dict(model_data, strict=False)

    file_path = ".data/%s_data.tar.xz"
    dataset = dto.wds_load(file_path % "train", limit=limit)
    # dataset = dto.apply(dataset, pad_audio)
    # dataset = dto.apply(dataset, limit_file)
    # print(dataset[0])
    # print(dataset[1])
    # dataset = [d for d in dataset if "SA" not in d.source]  # loss is larger than mean ms error
    # dataset = [d for d in dataset if "SA" in d.source]
    print(len(dataset))
    dataset_train = UtteranceDataset(dataset)
    for batch in UtteranceDataset(dataset).batch(64):
        print(batch)
        break
    # dataset_test = UtteranceDataset( dto.apply(dto.wds_load(file_path % "test", limit=512), pad_audio))

    trainer.fit(module, dataset_train.batch(64))
    time.sleep(2)
    trainer.save_checkpoint(checkpoint)

    # for dataset in dataset_train, dataset_test:
    #     generated = []
    #     for batch in dataset.batch(128, shuffle=False):
    #         generated.extend(module.with_gradient(batch).original)
    #     # result = trainer.predict(module.with_gradient, dataloaders=dataset.batch(128, shuffle=False))
    #     # print(result)
    #     # generated = sum([r.original for r in result], start=[])
    #     # print(len(generated))
    #     # generated = fix_borders(generated, report_error=550)
    #     report_borders(generated, plotting=False)
    #     # break
