from typing import List, Optional, NamedTuple
import os

import torch
from torch.utils.data import Dataset, DataLoader

from spn.dto.base import DataTransferObject
from spn.models.soft_pointer_network import SoftPointerNetwork
from spn.tools import display_diff
import numpy as np
import pandas as pd

import pyloudnorm as pyln
import soundfile as sf
from python_speech_features import logfbank
from spn.constants import (
    MAP_LABELS,
    MERGE_DOUBLES,
    NO_BORDER_MAPPING,
    TRANSFORM_MAPPING, WIN_SIZE, WIN_STEP, INPUT_SIZE,
)

from tqdm.auto import tqdm
import torch.nn as nn


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


class File(DataTransferObject):
    source: tuple
    config: tuple
    features_spectogram: Array[float]
    features_phonemes: Array[float]
    target_timestamps: Array[float]
    output_timestamps: Optional[Array[float]] = None


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
            end_ms = (current + end_ms) / 2  # TODO: quiet should be included in the current?

        if current >= end_ms:
            continue

        unholy_combination = prev in NO_BORDER_MAPPING and tag in NO_BORDER_MAPPING
        if prev == tag and MERGE_DOUBLES or unholy_combination or q_tag:
            tag_id, _ems = tag_mapping[-1]
            tag_mapping[-1] = (tag_id, end_ms)

        else:
            tag_mapping.append((tag, end_ms))

        prev = tag  # handle same tag occurrence
        current = int(end_ms // ms_per_step + 1) * ms_per_step

    transcript = [
        np.array(MAP_LABELS[tag][0])
        for tag, end_ms in tag_mapping
    ]
    borders = [end_ms for tag, end_ms in tag_mapping]
    return borders, transcript


def load_files(base, files, winlen=WIN_SIZE, winstep=WIN_STEP, nfilt=INPUT_SIZE):
    rate = 16000
    meter = pyln.Meter(rate)  # create BS.1770 meter
    ms_per_step = winstep * 1000

    output = []
    for i, source in enumerate(tqdm(files)):
        label_file, audio_file = source
        label_file = os.path.join(base, "data", label_file)
        audio_file = os.path.join(base, "data", audio_file)

        borders, transcription = process_audio(label_file, ms_per_step)

        audio, read_rate = sf.read(audio_file)
        assert read_rate == rate, f"{read_rate} != {rate}"

        # loudness = meter.integrated_loudness(audio)
        # audio = pyln.normalize.loudness(audio, loudness, -40.0)

        fbank_feat = logfbank(
            audio,
            rate,
            winlen=winlen,
            winstep=winstep,
            nfilt=nfilt,
        )  # TODO: remove scaling

        # some audio instances are too short for the audio transcription
        # and the winlen cut :(
        # fbank_feat = np.vstack([fbank_feat] + [fbank_feat[-1]] * 10)
        # fbank_feat = fbank_feat[:int(borders[-1] // ms_per_step)]

        output.append(File(
            source=source,
            config=(winlen, winstep),
            features_spectogram=torch.FloatTensor(fbank_feat),
            features_phonemes=torch.FloatTensor(transcription),
            target_timestamps=torch.FloatTensor(borders),
        ))

    return output


def fix_borders(
    generated: List[File], report_error=300
):
    output = []
    for item in generated:
        borders = item.output_timestamps.copy()

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

        prev = 0
        for i, v in enumerate(borders):
            assert v >= prev, f"This should never happen! {i}"
            prev = v

        diff = item.target_timestamps - item.output_timestamps
        if np.abs(diff).max() > report_error:
            print(f"[id:{item.source}]  [{diff.min():5.0f} {diff.max():5.0f}]  {switched}")

    return output


def plot(audio, transcript):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    f, axarr = plt.subplots(2, figsize=(8, 8))
    axarr[0].imshow(audio.T, origin="lower", aspect="auto", cmap=cm.winter)
    # axarr[1].imshow(torch.load("vanilla_audio.pth").cpu().T, origin="lower", aspect="auto", cmap=cm.winter)
    # axarr[1].imshow(torch.load("vanilla_transcription.pth").cpu().T, origin="lower", aspect="auto", cmap=cm.winter)
    axarr[1].imshow(transcript.T, origin="lower", aspect="auto", cmap=cm.winter)

    # print((torch.load("vanilla_audio.pth").cpu().T - audio.T).sum(), "FUMMM")
    plt.title('NEW IMPROVED')
    plt.show()


def generate_borders(
    model, dataset, batch_size: int = 32
):
    model.eval()

    winlen, winstep = dataset.files[0].config
    ms_per_step = winstep * 1000
    output = []
    for batch in tqdm(dataset.batch(batch_size)):
        features_audio = batch['features_spectogram']
        features_transcription = batch['features_phonemes']

        # print(batch['source'])
        # print("features_transcription", features_transcription.padded.shape, features_transcription.masks.sum())
        # print("features_audio", features_audio.padded.shape, features_audio.masks.sum())
        # plot(features_audio.padded[0], features_transcription.padded[0])
        borders_predicted = model(
            features_transcription.padded.to(model.device),
            features_transcription.masks.to(model.device),
            features_audio.padded.to(model.device),
            features_audio.masks.to(model.device),
        ).cpu().detach().numpy()

        for i, item in enumerate(batch['original']):
            item: File
            length = batch['target_timestamps'].lengths[i]
            predicted_border = borders_predicted[i, :length]
            output.append(item.update(output_timestamps=predicted_border * ms_per_step))

    return output


def report_borders(
    generated: List[File], plotting=False
):
    # item = generated[0]
    # print("truth", item.target_timestamps.round().tolist())
    # print("gen  ", item.output_timestamps.round().tolist())
    diffs = [item.target_timestamps[:-1] - item.output_timestamps[:-1] for item in generated]
    diff = np.concatenate(diffs)
    # print((torch.load("diff.pth") - diff).sum(), "FUMMM")

    print("TOTAL", np.abs(diff).sum(), diff.shape)
    display_diff(diff, "position", plotting=plotting)
    return generated


class MyCustomDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __getitem__(self, index):
        return self.files[index]

    def __len__(self):
        return len(self.files)

    def features_batch_process(self, batch_features: List[np.array]) -> UtteranceBatch:
        # this is used when a list of data items is transformed into a batch
        # TODO: could we, should we use pack_padded_sequence
        padded = nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
        lens = torch.tensor([len(item) for item in batch_features])
        _b, max_len, *_f = padded.shape
        return UtteranceBatch(
            padded,
            torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1),
            lens,
        )

    def collate_fn(self, batch: List[DataTransferObject]):
        result = {'original': batch}
        first = batch[0]
        for k in first.__fields__.keys():
            values = [getattr(item, k) for item in batch]
            result[k] = self.features_batch_process(values) if torch.is_tensor(values[0]) else values
        return result

    def batch(self, batch_size):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

limit = 128  # None means unlimited, else 100 would mean to load only the first 100 files
# limit = None

base = ".data"
file_path = f"{base}/test_data.csv"
test_files = load_csv(file_path, sa=False)[:limit]
# test_files = [(a, b) for a, b in test_files if 'TEST/DR1/FELC0/SX126' in a]
# test_files = [(a, b) for a, b in test_files if 'TEST/DR1/MREB0/SI2005' in a]
test_files = load_files(base, test_files)
print(len(test_files))

test_dataset = MyCustomDataset(test_files)

soft_pointer_model = SoftPointerNetwork(54, 26, 256, device=device)
soft_pointer_model.load(path="spn/trained_weights/position_model-final.pth")

generated = generate_borders(soft_pointer_model.with_gradient, test_dataset)
generated = fix_borders(generated)
report_borders(generated, plotting=False)
