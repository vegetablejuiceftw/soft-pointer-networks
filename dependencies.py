import contextlib
import cProfile
import gc
import json
import math
import os
import random
import sys
import uuid
from collections import Counter, OrderedDict, defaultdict, namedtuple
from itertools import chain
from os.path import join
from pprint import pprint
from random import Random, choice, sample
from time import sleep
from typing import List, NamedTuple

import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import Audio, Image, display
from matplotlib import cm, gridspec
from matplotlib.pyplot import figure
from matplotlib.ticker import FormatStrFormatter
from numpy import dot
from numpy.linalg import norm
from torch.utils.data import DataLoader, Dataset

import librosa
import pandas as pd
import pyrubberband as pyrb
import soundfile as sf
from dtaidistance import dtw as dtaidtw
from dtaidistance.dtw_ndim import warping_paths
from dtw import accelerated_dtw as adtw
from dtw import dtw
from fastdtw import dtw as slowdtw
from fastdtw import fastdtw
from google.colab import drive
from python_speech_features import logfbank, mfcc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchtext.data import BucketIterator, Field, RawField
