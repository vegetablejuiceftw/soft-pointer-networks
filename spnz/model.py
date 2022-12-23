from random import randint
from typing import Dict

import torch
from torch import nn
import pytorch_lightning as pl

from spn.models.base import ModeSwitcherBase, ExportImportMixin
from spn.models.panns import Cnn14_DecisionLevelAtt, Spectogram, SpectogramCNN
