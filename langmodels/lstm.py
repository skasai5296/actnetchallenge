import sys, os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from utils import *



class Captioning(nn.Module):
    def __init__(self,
                 method='LSTM',
                 vocab_size=None,
                 mem_dim=None
                 ):
        super(Captioning, self).__init__()
        self.method = method
        self.vocab_size = vocab_size
        self.mem_dim = mem_dim
        if method = 'LSTM':
            self.model = nn.Sequential(nn.LSTM())

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
