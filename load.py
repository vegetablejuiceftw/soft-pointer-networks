from typing import List, Optional, NamedTuple, Dict, Tuple
import os

import pandas as pd
import torch

import numpy as np
from pydantic import BaseModel

from tqdm.auto import tqdm

import pyloudnorm as pyln
import soundfile as sf
from python_speech_features import logfbank
from spn.constants import (
    MAP_LABELS,
    NO_BORDER_MAPPING,
    TRANSFORM_MAPPING, WIN_SIZE, WIN_STEP, INPUT_SIZE, WEIGHTS,
)

class UtteranceBatch(NamedTuple):
    padded: torch.tensor
    masks: torch.tensor
    lengths: torch.tensor


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


class File(BaseModel):
    source: tuple
    config: tuple
    ids_phonemes: Tuple[str, ...]
    weights_phonemes: Array[float]
    features_spectogram: Array[float]
    features_phonemes: Array[float]
    target_timestamps: Array[float]
    output_timestamps: Optional[Array[float]] = None

    def update(self, **kwargs):
        return self.copy(update=kwargs)


def load_csv(file_path, sa=False):
    """Filters out the files names of phonetic and sound data in pairs."""
    df = pd.read_csv(file_path, delimiter=",", nrows=None)
    df = df.sort_values(by=["path_from_data_dir"])
    audio_mask = df.is_audio & df.is_converted_audio
    phn_mask = df.filename.str.contains(".PHN")
    sa_mask = df.filename.str.contains("SA") == False  # noqa # pylint: disable=singleton-comparison
    df = df.loc[audio_mask | phn_mask]
    if not sa:
        df = df.loc[sa_mask]
    a = df.loc[phn_mask.fillna(False)].path_from_data_dir
    b = df.loc[audio_mask].path_from_data_dir
    assert len(a) == len(b)
    return list(zip(a, b))


def process_audio(label_file: str, ms_per_step: float):
    with open(label_file) as f:
        lines = list(f.readlines())
        labels = []
        sample_rate = 16000
        for line in lines:
            _, end, tag = line.split()
            end_sec = float(end) / sample_rate
            labels.append((end_sec * 1000, tag))

    tag_mapping = []
    current, prev = 0, None
    for end_ms, tag in labels:
        tag = TRANSFORM_MAPPING.get(tag, tag)
        q_tag = False  # /q/ tag
        if tag is None:
            tag = prev
            q_tag = True

        unholy_combination = prev in NO_BORDER_MAPPING and tag in NO_BORDER_MAPPING
        if prev == tag or unholy_combination or q_tag:
            tag_id, _ems = tag_mapping[-1]
            tag_mapping[-1] = (tag_id, end_ms)
        else:
            tag_mapping.append((tag, end_ms))

        prev = tag  # handle same tag occurrence

    ids, borders, transcript = zip(*[
        (tag, end_ms, np.array(MAP_LABELS[tag][0]))
        for tag, end_ms in tag_mapping
    ])
    weights = tuple(WEIGHTS[tag] for tag in ids)
    return ids, borders, transcript, weights


def load_files(base, files, winlen=WIN_SIZE, winstep=WIN_STEP, nfilt=INPUT_SIZE):
    rate = 16000
    meter = pyln.Meter(rate)  # create BS.1770 meter
    ms_per_step = winstep * 1000

    output = []
    for i, source in enumerate(tqdm(files)):
        label_file, audio_file = source
        label_file = os.path.join(base, "data", label_file)
        audio_file = os.path.join(base, "data", audio_file)

        transcription_ids, borders, transcription, weights = process_audio(label_file, ms_per_step)

        audio, read_rate = sf.read(audio_file)
        assert read_rate == rate, f"{read_rate} != {rate}"

        loudness = meter.integrated_loudness(audio)
        audio = pyln.normalize.loudness(audio, loudness, -40.0)

        fbank_feat = logfbank(
            audio,
            rate,
            winlen=winlen,
            winstep=winstep,
            nfilt=nfilt,
        )  # TODO: remove scaling

        # some audio instances are too short for the audio transcription
        # and the winlen cut :(
        fbank_feat = np.vstack([fbank_feat] + [fbank_feat[-1]] * 20)
        fbank_feat = fbank_feat[:int(borders[-1] // ms_per_step)]

        output.append(File(
            source=source,
            config=(winlen, winstep),
            ids_phonemes=transcription_ids,
            weights_phonemes=torch.FloatTensor(weights),
            features_spectogram=torch.FloatTensor(fbank_feat),
            features_phonemes=torch.FloatTensor(transcription),
            target_timestamps=torch.FloatTensor(borders) / ms_per_step,
        ))

    return output
