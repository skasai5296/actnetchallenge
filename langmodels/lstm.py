import sys, os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append(os.pardir)
from utils.utils import *


# embedding module.
# (***) -> (***, embedding_size)
class Embedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size=512
                 ):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.emb = nn.Embedding(self.vocab_size, self.embedding_size)

    def forward(self, x):
        return self.emb(x)

    # TODO
    def init_pretrained_weights(self, vocab):
        pass


# captioning module.
# lstm_memory must be same as output dimension of encoder (3DCNN). TCHW must be flattened.
class RNNCaptioning(nn.Module):
    def __init__(self,
                 method='LSTM',
                 ft_size=512,
                 emb_size=512,
                 lstm_memory=512,
                 vocab_size=100,
                 max_seqlen=20,
                 num_layers=1,
                 dropout_p=0.5
                 ):
        super(RNNCaptioning, self).__init__()
        self.method = method
        self.ft_size = ft_size
        self.emb_size = emb_size
        self.lstm_memory = lstm_memory
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen
        self.num_layers = num_layers

        self.linear1 = nn.Linear(self.ft_size, self.emb_size)
        self.emb = nn.Embedding(self.vocab_size, self.emb_size)
        if method == 'LSTM':
            #self.rnn = nn.LSTM(self.emb_size, self.lstm_memory, self.num_layers, batch_first=True)
            self.rnn = nn.LSTMCell(self.emb_size, self.lstm_memory)
        self.linear2 = nn.Linear(self.lstm_memory, self.vocab_size)
        self.dropout = nn.Dropout(p=dropout_p)

    def init_embedding(self, tensor):
        self.emb.weight.data.copy_(tensor)

    # THWC must be flattened for image feature
    # image_feature : (batch_size, ft_size)
    # captions : (batch_size, seq_len) (LongTensor of indexes, padded sequence)
    # lengths : (batch_size,) (LongTensor of lengths)
    # returns : list of tensors of size (batch_size, vocab_size), list of ints (lengths) to backward for each caption
    def forward(self, image_feature, captions, lengths, init_state=None):
        # feature : (batch_size, emb_size)
        feature = self.linear1(image_feature)
        # h_n, c_n : (batch_size, lstm_memory)
        h_n, c_n = self.rnn(feature, init_state)
        # token1 : (batch_size, vocab_size)
        token1 = self.linear2(h_n)
        outputlist = [token1]
        # captions : (seq_len, batch_size, emb_size)
        captions = self.emb(captions).transpose(0, 1)
        idx = 0
        maxlen = min(self.max_seqlen, captions.size(0))
        while idx < maxlen-1:
            # h_n : (batch_size, lstm_memory)
            (h_n, c_n) = self.rnn(captions[idx], (h_n, c_n))
            # tokenn : (batch_size, vocab_size)
            tokenn = self.dropout(self.linear2(h_n))
            outputlist.append(tokenn)
            idx += 1
        # outcaption : would be tensor of size (batch_size, maxlen, vocab_size)
        outcaption = torch.stack(outputlist).transpose(0, 1)
        lengths = [min(length, self.max_seqlen) for length in lengths]
        outputs = pack_padded_sequence(outcaption, lengths, batch_first=True)
        return outcaption, lengths

    # image_feature : (batch_size, ft_size)
    # method : one of ['greedy', 'beamsearch']
    def decode(self, image_feature, method='greedy', init_state=None):
        outputlist = []
        # feature : (batch_size, emb_size)
        feature = self.linear1(image_feature)
        # h_n, c_n : (batch_size, lstm_memory)
        h_n, c_n = self.rnn(feature, init_state)
        for idx in range(self.max_seqlen):
            # o_n : (batch_size, vocab_size)
            o_n = self.linear2(h_n)
            # tokenid : (batch_size,)
            tokenid = o_n.argmax(dim=1)
            outputlist.append(tokenid)
            # inputs : (batch_size, emb_size)
            inputs = self.emb(tokenid)
            # h_n : (batch_size, lstm_memory)
            (h_n, c_n) = self.rnn(inputs, (h_n, c_n))
        # outcaption : would be tensor of size (batch_size, max_seqlen)
        outcaption = torch.stack(outputlist).transpose(0, 1)
        return outcaption

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



