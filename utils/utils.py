import sys, os

sys.path.append(os.pardir)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from langmodels.vocab import tokenize


def collater(datas, token_level=False):
    # sort data by caption lengths for packing
    datas.sort(key=lambda x: x[1].size(0), reverse=True)
    clips, captions = zip(*datas)
    maxlen = datas[0][1].size(0)
    batchsize = len(captions)
    ten = []
    lengths = torch.tensor([cap.size(0) for cap in captions], dtype=torch.long)
    padded_captions = torch.zeros(batchsize, maxlen, dtype=torch.long)
    for i, caption in enumerate(captions):
        length = caption.size(0)
        padded_captions[i, :length] = caption[:length]

    batched_clip = torch.stack(clips)
    return batched_clip, padded_captions, lengths

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
