import torch
import torchlibrosa as tl
import numpy as np
import matplotlib.pyplot as plt

hop_length = 128
spectrogram_extractor = tl.stft.Spectrogram(n_fft=2048, hop_length=hop_length)
logmel_extractor = tl.stft.LogmelFilterBank(sr=16000, n_fft=2048, n_mels=128)


def spectogram(audio: np.ndarray) -> torch.FloatTensor:
    audio = torch.FloatTensor(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    sp = spectrogram_extractor.forward(audio)
    logmel = logmel_extractor.forward(sp)
    return logmel.squeeze(1).squeeze(0).T


def show_tensor(tensor, index=1, figsize=(16, 16), limit=True, normalize=False):
    if len(tensor.shape) > 2:
        tensor = tensor[index]

    if hasattr(tensor, 'detach'): tensor = tensor.detach()
    if hasattr(tensor, 'cpu'): tensor = tensor.cpu()
    if hasattr(tensor, 'numpy'): tensor = tensor.numpy()
    if normalize:
        tensor -= tensor.min()
        tensor /= tensor.max()
    plt.figure(figsize=figsize)
    plt.title(str(tensor.shape))
    plt.imshow(tensor, cmap='winter')
    if limit:
        plt.clim(0, 1)
    plt.show()
