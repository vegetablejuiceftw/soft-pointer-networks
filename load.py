from typing import List, Optional, NamedTuple, Dict, Tuple
import os

import pandas as pd
import torch

import numpy as np
from pydantic import BaseModel

from tqdm.auto import tqdm
import webdataset as wds

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
    # features_spectogram: Array[float]
    features_audio: Array[float]
    features_phonemes: Array[float]
    target_timestamps: Array[float]
    target_durations: Array[float]
    # target_occurrences: Array[int]

    output_timestamps: Optional[Array[float]]
    output_durations: Optional[Array[float]]
    # output_occurrences: Optional[Array[int]]
    # audio: Optional[Array[int]]

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


def process_audio(label_file: str, sample_rate=16000):
    with open(label_file) as f:
        lines = list(f.readlines())
        labels = []
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
    expected_rate = 16000
    # meter = pyln.Meter(rate)  # create BS.1770 meter
    # ms_per_step = winstep * 1000

    output = []
    for i, source in enumerate(tqdm(files)):
        label_file, audio_file = source
        label_file = os.path.join(base, "data", label_file)
        audio_file = os.path.join(base, "data", audio_file)

        transcription_ids, borders, transcription, weights = process_audio(label_file)

        audio, read_rate = sf.read(audio_file)
        assert read_rate == expected_rate, f"{read_rate} != {expected_rate}"

        # loudness = meter.integrated_loudness(audio)
        # audio = pyln.normalize.loudness(audio, loudness, -40.0)

        # fbank_feat = logfbank(
        #     audio,
        #     rate,
        #     winlen=winlen,
        #     winstep=winstep,
        #     nfilt=nfilt,
        # )  # TODO: remove scaling
        #
        # # some audio instances are too short for the audio transcription
        # # and the winlen cut :(
        # max_len = int(borders[-1] // ms_per_step) + 8
        # fbank_feat = np.vstack([fbank_feat] + [fbank_feat[-1]] * 40)
        # fbank_feat = fbank_feat[:max_len]

        # target_occurrence = np.zeros(max_len, dtype=np.int64)
        # current = 0
        # for phoneme, border in zip(transcription_ids, borders):
        #     target_occurrence[current:] = MAP_LABELS[phoneme][1]
        #     current = int(border // ms_per_step)

        durations = np.array(borders) / borders[-1]
        durations[1:] -= durations[:-1]

        output.append(File(
            source=source,
            config=(0, 0),
            ids_phonemes=transcription_ids,
            weights_phonemes=np.array(weights, dtype=np.float32),
            # features_spectogram=np.array(fbank_feat, dtype=np.float32),
            features_phonemes=np.array(transcription, dtype=np.float32),
            target_timestamps=np.array(borders, dtype=np.float32),
            target_durations=np.array(durations, dtype=np.float32),
            # target_occurrences=target_occurrence,
            features_audio=np.array(audio, dtype=np.float32),
        ))

    return output


def wds_load(file_path, limit=None) -> List[File]:
    files = list(tqdm(wds.WebDataset(file_path).slice(limit).decode().map(lambda data: File.construct(**data["data.npz"]))))
    return files


if __name__ == '__main__':
    import time

    limit = None
    # limit = 32

    for split in "test", "train":
        base = ".data"
        files = load_csv(f"{base}/{split}_data.csv", sa=False)[:limit]
        files = load_files(base, files)

        file_path = f"{split}_data.tar.xz"
        with wds.TarWriter(file_path, compress=True) as dst:
            for file in files:
                item = {
                    "__key__": file.source[-1].replace(".wav", "").replace(".WAV", ""),
                    "data.npz": file.dict(exclude_unset=True),
                }
                dst.write(item)

    # arr = wds_load(file_path)
    # print(arr[0])

    # os.makedirs("dumps", exist_ok=True)
    # total_floats = sum(np.multiply(*f.features_spectogram.shape) for f in files)
    # print(f"Total floats: {total_floats} from {len(files)} files")
    #
    # compress = False
    # with wds.TarWriter("dumps/webdataset.npy.tar.xz", compress=compress) as dst:
    #     for file in files:
    #         item = {
    #             "__key__": file.source[-1],
    #             "features_spectogram.npy": file.features_spectogram.numpy(),
    #         }
    #         dst.write(item)
    #
    # with wds.TarWriter("dumps/webdataset.npz.tar.xz", compress=compress) as dst:
    #     for file in files:
    #         item = {
    #             "__key__": file.source[-1],
    #             "features_spectogram.npz": dict(array=file.features_spectogram),
    #         }
    #         dst.write(item)
    #
    # with wds.TarWriter("dumps/webdataset.pyd.tar.xz", compress=compress) as dst:
    #     for file in files:
    #         item = {
    #             "__key__": file.source[-1],
    #             "features_spectogram.pyd": file.features_spectogram,
    #         }
    #         dst.write(item)
    #
    # with wds.TarWriter("dumps/webdataset.pth.tar.xz", compress=compress) as dst:
    #     for file in files:
    #         item = {
    #             "__key__": file.source[-1],
    #             "features_spectogram.pth": file.features_spectogram,
    #         }
    #         dst.write(item)
    #
    #
    # def get_size(path):
    #     size = os.path.getsize(path)
    #     if size < 1024:
    #         return f"{size} bytes"
    #     elif size < 1024 * 1024:
    #         return f"{round(size / 1024, 2)} KB"
    #     elif size < 1024 * 1024 * 1024:
    #         return f"{round(size / (1024 * 1024), 2)} MB"
    #     elif size < 1024 * 1024 * 1024 * 1024:
    #         return f"{round(size / (1024 * 1024 * 1024), 2)} GB"
    #
    #
    # for file_name in os.listdir('dumps'):
    #     file_path = 'dumps/' + file_name
    #
    #     start = time.time()
    #     for _ in range(10):
    #         if "webdataset" in file_name:
    #             arr = list(wds.WebDataset(file_path).decode())
    #     e = arr[0]
    #     e.pop("__key__")
    #     e.pop("__url__")
    #     e = next(iter(e.values()))
    #     total_time = f'   Time: {time.time() - start:.4f}   {type(e).__name__}'
    #     print(file_name, "   ", get_size(path=file_path), total_time)
