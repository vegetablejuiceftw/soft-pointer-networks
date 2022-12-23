from random import choice
from time import sleep
from typing import List, Dict

import numpy
import torch
from torch.utils.data import Dataset, DataLoader

from spn.models.base import ModeSwitcherBase
from spn.models.soft_pointer_network import SoftPointerNetwork
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import torch.nn as nn


def detach(tensor) -> numpy.ndarray:
    if hasattr(tensor, 'detach'): tensor = tensor.detach()
    if hasattr(tensor, 'cpu'): tensor = tensor.cpu()
    if hasattr(tensor, 'numpy'): tensor = tensor.numpy()
    tensor = tensor.copy()
    return tensor

