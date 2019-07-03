import sys, os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#from utils.utils import *


# captioning module.
# lstm_memory must be same as output dimension of encoder (3DCNN). TCHW must be flattened.
class LSTMCaptioning(nn.Module):
    def __init__(self,
                 ft_size=512,
                 emb_size=512,
                 lstm_memory=512,
                 vocab_size=100,
                 max_seqlen=20,
                 num_layers=1,
                 dropout_p=0.1
                 ):
        super(RNNCaptioning, self).__init__()
        self.ft_size = ft_size
        self.emb_size = emb_size
        self.lstm_memory = lstm_memory
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen
        self.num_layers = num_layers

        self.linear1 = nn.Linear(self.ft_size, self.emb_size)
        self.emb = nn.Embedding(self.vocab_size, self.emb_size)
        self.rnn = nn.LSTM(self.emb_size, self.lstm_memory, self.num_layers, batch_first=True, dropout=dropout_p)
        #self.rnn = nn.LSTMCell(self.emb_size, self.lstm_memory)
        self.linear2 = nn.Linear(self.lstm_memory, self.vocab_size)

    def init_embedding(self, tensor):
        self.emb.weight.data.copy_(tensor)

    # THWC must be flattened for image feature
    # image_feature : (batch_size, ft_size)
    # captions : (batch_size, seq_len)
    # lengths : (batch_size,) (LongTensor of lengths)
    # returns : list of tensors of size (batch_size, vocab_size), list of ints (lengths) to backward for each caption
    def forward(self, image_feature, captions, lengths, init_state=None):
        # feature : (batch_size, 1, emb_size)
        feature = self.linear1(image_feature).unsqueeze(1)
        # captions : (batch_size, seq_len, emb_size)
        captions = self.emb(captions)
        # inseq : (batch_size, 1+seq_len, emb_size)
        inseq = torch.cat((feature, captions), dim=1)
        packed = pack_padded_sequence(inseq, lengths, batch_first=True)
        hiddens, _ = self.rnn(packed)
        outputs = self.linear2(hiddens[0])
        return outputs

    # image_feature : (batch_size, ft_size)
    # method : one of ['greedy', 'beamsearch']
    def sample(self, image_feature, method='greedy', init_state=None):
        outputlist = []
        # feature : (batch_size, emb_size)
        inputs = self.linear1(image_feature).unsqueeze(1)
        for idx in range(self.max_seqlen):
            hiddens, states = self.rnn(inputs, init_state)
            outputs = self.linear2(hiddens.squeeze(1))
            _, pred = outputs.max(1)
            outputlist.append(pred)
            inputs = self.emb(pred)
            inputs = inputs.squeeze(1)
        # outcaption : would be tensor of size (batch_size, max_seqlen)
        sampled = torch.stack(outputlist, 1)
        return sampled

    # TODO
    def init_pretrained_weights(self, vocab):
        self.emb.init_pretrained_weights(vocab)




class Transformer(nn.Module):
    def __init__(self,
                 vocab_size=None,
                 mem_dim=None
                 ):
        super(Transformer, self).__init__()
        self.model = None

    def forward(self, x):
        pass



if __name__ == '__main__':
    bs = 10
    data = torch.randn(bs, 512)
    captionidx = torch.randint(0, 100, size=(bs, 20))
    model = RNNCaptioning()
    print(model(data, captionidx).size())



