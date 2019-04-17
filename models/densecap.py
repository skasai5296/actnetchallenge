import sys, os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from utils import *
from dataset.activitynet_captions import ActivityNetCaptions


class DenseCap(nn.Module):
    def __init__(self,
                 vocab_size=None
                 mem_dim=1024
                 ):
        super(DenseCap, self).__init__()
        self.model = None

    def forward(self, x):
        pass


class Captioning(nn.Module):
    def __init__(self,
                 vocab_size=None,
                 mem_dim=None
                 ):
        super(Captioning, self).__init__()
        self.model = None

    def forward(self, x):
        pass


class LSTM(nn.Module):
    def __init__(self,
                 vocab_size=None,
                 mem_dim=None
                 ):
        super(LSTM, self).__init__()
        self.model = None

    def forward(self, x):
        pass


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size=None,
                 mem_dim=None
                 ):
        super(Transformer, self).__init__()
        self.model = None

    def forward(self, x):
        pass
