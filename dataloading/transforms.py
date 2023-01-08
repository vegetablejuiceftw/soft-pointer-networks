from dataclasses import dataclass
from typing import List, Callable, Tuple

import pyloudnorm as pyln
import numpy as np

from dataloading import dto


meter = pyln.Meter(16000)  # create BS.1770 meter


def normalize_loudness(file: dto.File):
    audio = file.audio
    loudness = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loudness, -40.0)
    return file.update(audio=audio)


def shift_timeline(file: dto.File, delta_ms):
    return file.update(
        phonetic_detail=file.phonetic_detail.update(
            start=file.phonetic_detail.start + delta_ms,
            stop=file.phonetic_detail.stop + delta_ms,
        ),
        word_detail=file.word_detail.update(
            start=file.word_detail.start + delta_ms,
            stop=file.word_detail.stop + delta_ms,
        ),
    )


@dataclass
class PadAudio:
    start_ms: int = 0
    end_ms: int = 513

    def handle(self, file: dto.File):
        if self.start_ms:
            file = shift_timeline(file, self.start_ms)
        return file.update(audio=np.pad(file.audio, [self.start_ms * file.msr, self.end_ms * file.msr]))


def limit_timeline(detail: dto.Timeline, start_ms, end_ms):
    start_index = 0
    for index, (start, stop) in enumerate(zip(detail.start, detail.stop)):
        if start < start_ms:
            start_index = index

        if stop > end_ms:
            detail = detail.update(
                utterance=detail.utterance[start_index:index],
                start=detail.start[start_index:index],
                stop=detail.stop[start_index:index],
            )
            if detail.id is not None:
                detail = detail.update(id=detail.id[start_index:index])
            break
    return detail


def limit_file(file: dto.File, start_ms, end_ms):
    if start_ms:
        # we remove audio from the start, so time shift to the left
        file = shift_timeline(file, -start_ms)

    return file.update(
        audio=file.audio[start_ms * file.msr:end_ms * file.msr],
        phonetic_detail=limit_timeline(file.phonetic_detail, start_ms, end_ms),
        word_detail=limit_timeline(file.word_detail, start_ms, end_ms),
    )


@dataclass
class SliceTimeline:
    start_ms: int = 0
    end_ms: int = 1000

    def handle(self, file: dto.File):
        file = limit_file(file, self.start_ms, self.end_ms)
        if not file.phonetic_detail.utterance:
            return None
        return file


@dataclass
class DefaultTransform:
    transforms: Tuple[Callable[[dto.File], dto.File], ...] = (
        SliceTimeline(end_ms=1000).handle,
        normalize_loudness,
        PadAudio(end_ms=132).handle,
    )

    def handle(self, file: dto.File):
        for f in self.transforms:
            file = f(file)
            if not file or not file.phonetic_detail.utterance:
                return None
        return file
