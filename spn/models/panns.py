import logging

import numpy as np
import torch


from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from torch import nn
from torch.nn import functional as F

from panns_inference.pytorch_utils import do_mixup, interpolate, pad_framewise_output, move_data_to_device
from panns_inference.models import ConvBlock, init_layer, init_bn
from panns_inference import AudioTagging, SoundEventDetection, labels


def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.
    Args:
      clipwise_output: (classes_num,) i.e no batch data
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(min(len(sorted_indexes), 10)):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
                                  clipwise_output[sorted_indexes[k]]))


class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()

        self.activation = activation
        self.temperature = temperature
        # self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)

        # self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()

    def init_weights(self):
        # init_layer(self.att)
        init_layer(self.cla)
        # init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        # norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        # x = torch.sum(norm_att * cla, dim=2)
        return None, None, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad * 0), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


class Cnn14_DecisionLevelAtt(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, interpolate_ratio=32, dropout=0.35):

        super(Cnn14_DecisionLevelAtt, self).__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.reduction_ratio = 4
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

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=8,
                                               freq_drop_width=8, freq_stripes_num=8)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        # self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(1024, 2048, bias=True)
        self.att_block = AttBlock(2048, classes_num, activation='linear')

        self.init_weight()
        self.upsample = nn.Upsample(scale_factor=self.interpolate_ratio, mode='linear')

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    @property
    def ms_per_step(self):
        return self.hop_size / self.sample_rate * (self.reduction_ratio / self.interpolate_ratio) * 1000

    def interpolate(self, audio):
        if self.interpolate_ratio > 1:
            return self.upsample(audio.transpose(1, 2)).transpose(1, 2)
        return audio

    def forward(self, input, interpolate=True):
        """
        Input: (batch_size, data_length)"""

        dropout_cnn = self.dropout
        dropout = self.dropout

        with torch.no_grad():
            audio_samples = input.shape[-1]
            multiple_of = (audio_samples // self.hop_size // self.reduction_ratio + 3) * self.hop_size * self.reduction_ratio - 1
            input = pad_framewise_output(input.unsqueeze(-1), multiple_of).squeeze(-1)

            x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

            if self.training:
                x = self.spec_augmenter(x)

            spectogram = x
            # return {'spectogram': spectogram.squeeze(1)}

            logging.debug("x1 %s", x.shape)
            x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=dropout_cnn, training=self.training)
            x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=dropout_cnn, training=self.training)

        x = self.conv_block3(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=dropout_cnn, training=self.training)

        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=dropout_cnn, training=self.training)

        x = self.conv_block5(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=dropout_cnn, training=self.training)

        # x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        # x = F.dropout(x, p=dropout_cnn, training=self.training)
        x = torch.mean(x, dim=3)
        logging.debug("x2 %s", x.shape)

        # x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        # x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        # x = x1 + x2
        x = F.dropout(x, p=dropout, training=self.training)
        x = x.transpose(1, 2)
        logging.debug("x3 %s", x.shape)

        # # # #
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=dropout, training=self.training)
        logging.debug("x4 %s", x.shape)

        _, _, segmentwise_output = self.att_block(x)
        framewise_output = segmentwise_output.transpose(1, 2)

        if interpolate:
            framewise_output = self.interpolate(framewise_output)

        # logging.info("framewise_output %s %s", framewise_output.shape, spectogram.shape)
        # framewise_output = pad_framewise_output(framewise_output, frames_num)
        # logging.info("framewise_output %s", framewise_output.shape)

        return {
            'framewise_output': framewise_output,
            'spectogram': spectogram.squeeze(1),
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

