import torch


from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from torch import nn
from torch.nn import functional as F

from panns_inference.models import ConvBlock


class SpectogramCNN(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, interpolate_ratio=32, dropout=0.35):

        super().__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.reduction_ratio = 2 ** 2
        self.interpolate_ratio = interpolate_ratio
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.dropout = dropout

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)

        # TODO: might cause worse performance?
        # Spec augmenter
        # self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=8,
        #                                        freq_drop_width=8, freq_stripes_num=8)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.cla = nn.Conv1d(in_channels=512, out_channels=classes_num, kernel_size=1, stride=1, padding=0, bias=True)

        self.upsample = nn.Upsample(scale_factor=self.interpolate_ratio, mode='linear')

    @property
    def ms_per_step(self):
        return self.hop_size / self.sample_rate * (self.reduction_ratio / self.interpolate_ratio) * 1000

    def interpolate(self, audio):
        if self.interpolate_ratio > 1:
            return self.upsample(audio.transpose(1, 2)).transpose(1, 2)
        return audio

    def forward(self, input, interpolate=True):
        """ Input: (batch_size, data_length) """

        # with torch.no_grad():  # find_unused_parameters=True
        if True:
            x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

            # if self.training:
            #     x = self.spec_augmenter(x)

            # spectogram = x.squeeze(1)

            dropout_cnn = self.dropout

            x = self.conv_block1(x, pool_size=(2 if self.reduction_ratio >= 2 ** 1 else 1, 2), pool_type='avg')
            x = F.dropout(x, p=dropout_cnn, training=self.training)

            x = self.conv_block2(x, pool_size=(2 if self.reduction_ratio >= 2 ** 2 else 1, 2), pool_type='avg')
            x = F.dropout(x, p=dropout_cnn, training=self.training)

            x = self.conv_block3(x, pool_size=(2 if self.reduction_ratio >= 2 ** 3 else 1, 2), pool_type='avg')
            x = F.dropout(x, p=dropout_cnn, training=self.training)

            x = self.conv_block4(x, pool_size=(2 if self.reduction_ratio >= 2 ** 4 else 1, 2), pool_type='avg')
            x = F.dropout(x, p=dropout_cnn, training=self.training)

        # print(spectogram.shape, x.shape)

        x = torch.mean(x, dim=3)
        x = self.cla(x)

        # print(spectogram.shape, x.shape)
        framewise_output = x.transpose(1, 2)

        if interpolate:
            # spectogram = self.interpolate(spectogram)
            framewise_output = self.interpolate(x)

        # print(spectogram.shape, framewise_output.shape)
        return {
            'spectogram': framewise_output,
        }


if __name__ == '__main__':
    model = Cnn14_DecisionLevelAtt(
        sample_rate=32000, window_size=1024,
        hop_size=320, mel_bins=64, fmin=50, fmax=14000,
        classes_num=527)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.att_block.activation = "linear"

    model_data = torch.load('Cnn14_DecisionLevelAtt_mAP=0.425.pth', map_location=torch.device('cpu'))['model']
    # print(model_data.keys())
    # for key in tuple(model_data):
    #     if 'att_block' in key:
    #         del model_data[key]
    model.load_state_dict(model_data, strict=False)

