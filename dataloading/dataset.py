from typing import List, Dict, Union, Optional, Iterable

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import torch.nn as nn

from dataloading import dto


class UtteranceDataset(Dataset):
    def __init__(self, files: List[dto.File]):
        self.files = files
        # self.files = sorted(files, key=lambda x: -len(x.features_phonemes))

    def __getitem__(self, index):
        return self.files[index]

    def __len__(self):
        return len(self.files)

    def features_batch_process(self, batch_features: List[np.array]) -> dto.ArrayBatch:
        # this is used when a list of data items is transformed into a stacked batch
        # TODO: could we, should we, use pack_padded_sequence?
        padded = nn.utils.rnn.pad_sequence(
            [torch.tensor(e) for e in batch_features],
            batch_first=True, padding_value=0,
        )
        lens = torch.IntTensor([len(item) for item in batch_features])
        _b, max_len, *_f = padded.shape
        return dto.ArrayBatch(
            padded=padded,
            mask=torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1),
            length=lens,
        )

    def collate_dto(self, batch: List[dto.Base]):
        result = {}
        first = batch[0]
        for k in first.dict(exclude_unset=True).keys():
            values = [getattr(item, k) for item in batch]
            result[k] = values
            if getattr(values[0], 'dtype', None) in [np.float32, np.int64]:
                result[k] = self.features_batch_process(values)
            elif getattr(values[0], '__fields__', None):
                result[k] = self.collate_dto(values)
        return result

    def collate_fn(self, batch: List[dto.File]):
        batch = sorted(batch, key=lambda x: len(x.phonetic_detail.utterance))  # todo: reverse ordering?
        result = {'original': batch, **self.collate_dto(batch)}
        return dto.FileBatch.parse_obj(result)

    def batch(
        self, batch_size, shuffle=True, mp=True, num_workers=4, persistent_workers=True,
    ) -> Union[Iterable[dto.FileBatch], DataLoader]:
        extra = {}
        if mp:
            extra = dict(num_workers=num_workers, persistent_workers=persistent_workers, prefetch_factor=2)
        return DataLoader(
            dataset=self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn,
            pin_memory=True,
            **extra,
        )


if __name__ == '__main__':
    import os

    os.chdir(os.getcwd().split('/dataloading')[0])
    dataset = dto.wds_load(".data/test_data.tar.xz", limit=8)
    print("loaded")
    dataset = UtteranceDataset(dataset)
    print("start - batch")
    for batch in dataset.batch(batch_size=8, mp=False):
        print(batch)
        print(batch.audio.padded.shape)
        print(batch.phonetic_detail.stop.padded.shape)
        print(batch.phonetic_detail.id.padded.shape)
        print(batch.phonetic_detail.id.padded[0])
        print(batch.phonetic_detail.id.padded[-1])

        encoder_transcription_embedding = torch.nn.Embedding(40, 32, max_norm=True, padding_idx=0)
        result = encoder_transcription_embedding(batch.phonetic_detail.id.padded)
        print(result.shape)
        break
