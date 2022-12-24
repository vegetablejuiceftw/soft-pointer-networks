from copy import deepcopy

import numpy as np
from pydantic import BaseModel
import torch
from typing import List, Optional, NamedTuple, Dict, Tuple, Union
import webdataset as wds

from tqdm.auto import tqdm

COUNTS = {'pau': 48654, 'ih': 18347, 'n': 11874, 's': 10114, 'iy': 9663, 'l': 9451, 'r': 9064, 'ah': 8634, 'aa': 8293,
          'er': 7636, 'k': 6488, 't': 5899, 'm': 5600, 'ae': 5404, 'eh': 5293, 'z': 5046, 'd': 4793, 'w': 4379,
          'dh': 3879, 'dx': 3649, 'p': 3545, 'sh': 3259, 'ay': 3242, 'uw': 3213, 'f': 3128, 'ey': 3088, 'b': 3067,
          'ow': 2913, 'hh': 2836, 'g': 2772, 'v': 2704, 'y': 2349, 'ng': 1787, 'jh': 1581, 'ch': 1081, 'th': 1018,
          'oy': 947, 'aw': 945, 'uh': 756}


def apply(iterable, function):
    return [e for e in map(function, tqdm(iterable)) if e]


class ArrayMeta(type):
    def __getitem__(self, t):
        return type('Array', (Array,), {'__dtype__': t})


class Array(np.ndarray, metaclass=ArrayMeta):
    # TODO: maybe better solution for py3.9 https://github.com/samuelcolvin/pydantic/issues/380#issuecomment-736135687
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return val


class UtteranceBatch(NamedTuple):
    padded: torch.tensor
    masks: torch.tensor
    lengths: torch.tensor


class Base(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def update(self, **kwargs):
        return self.copy(update=kwargs)


class Timeline(Base):
    utterance: List[str]
    start: Array[float]
    stop: Array[float]

    @property
    def duration(self):
        return self.stop - self.start

    @property
    def duration_normalized(self):
        dur = self.duration
        return dur / self.stop[-1]


class File(Base):
    source: str
    sr: int
    audio: Array[float]

    phonetic_detail: Timeline
    word_detail: Timeline

    output_timestamps: Optional[Array[float]]
    output_durations: Optional[Array[float]]
    output_occurrences: Optional[Array[int]]

    @property
    def msr(self):
        """ ms sample rate """
        return self.sr // 1000


class ArrayBatch(Base):
    padded: Union[torch.IntTensor, torch.FloatTensor]
    mask:  torch.BoolTensor
    length: torch.IntTensor


class TimelineBatch(Base):
    utterance: List[List[str]]
    start: ArrayBatch
    stop: ArrayBatch


class FileBatch(Base):
    source: List[str]
    original: List[File]
    sr: List[int]
    audio: ArrayBatch

    phonetic_detail: TimelineBatch
    word_detail: TimelineBatch

    output_timestamps: Optional[torch.FloatTensor]
    output_durations: Optional[torch.FloatTensor]
    output_occurrences: Optional[torch.IntTensor]

    @property
    def length(self):
        return len(self.source)

    def __repr__(self):
        rep = str(self).replace('\n', ' ').replace('\t', '*')
        return f"<{rep}>"

    def __str__(self):
        # TODO: min/max/mean duration
        return f"FileBatch[{self.length}]" \
               f"\n\tAudio{(self.audio.length / self.original[0].msr).round().tolist()}" \
               f"\n\tPhoneme{self.phonetic_detail.stop.length.tolist()}" \
               f"\n\tWord{self.word_detail.stop.length.tolist()}" \
               f"\n\tOutput{[k for k in ['output_timestamps', 'output_durations', 'output_durations'] if getattr(self, k) is not None]}"


def convert(data: dict):
    data = deepcopy(data)
    for timeline in data['json']['phonetic_detail'], data['json']['word_detail']:
        timeline['start'] = np.array(timeline['start'], np.float32)
        timeline['stop'] = np.array(timeline['stop'], np.float32)

    return File(
        source=data['json']['source'],
        sr=data['json']['sampling_rate'],
        audio=data['audio.npy'],
        phonetic_detail=data['json']["phonetic_detail"],
        word_detail=data['json']["word_detail"],
    )


def wds_load(file_path, limit=None) -> List[File]:
    files = list(tqdm(wds.WebDataset(file_path).slice(limit).decode().map(convert)))
    return files


def write(dataset: list, destination: str, compress=False):
    with wds.TarWriter(destination, compress=compress) as dst:
        for item in tqdm(dataset):
            item = deepcopy(item)
            audio = item.pop('array')
            # phonetic_detail = item.pop('phonetic_detail')
            # word_detail = item.pop('word_detail')
            item = {
                "__key__": item['source'],
                "json": item,
                "audio.npy": audio,
                # "phonetic_detail.npz": phonetic_detail,
                # "word_detail.npz": word_detail,
            }
            dst.write(item)
