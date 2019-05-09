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


# captioning module BERT
class Bert(nn.Module):
    def __init__(self,
                 method='LSTM',
                 ft_size=512,
                 emb_size=512,
                 lstm_memory=512,
                 vocab_size=100,
                 max_seqlen=20,
                 num_layers=1
                 ):
        super(Bert, self).__init__()
    # THWC must be flattened for image feature
    # image_feature : (batch_size, ft_size)
    # x : (batch_size, seq_len) (LongTensor of indexes, padded sequence)
    # lengths : (batch_size,) (LongTensor of lengths)
    # returns : PackedSequence
    def forward(self, image_feature, x, lengths):
        seqlen = x.size(1)
        # h0 : (1, batch_size, lstm_memory)
        h0 = self.linear1(image_feature).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        # emb_seq : (batch_size, seq_len, emb_size)
        emb_seq = self.emb(x)
        # packed : (batch_size, seq_len, emb_size), PackedSequence
        packed = pack_padded_sequence(emb_seq, lengths, batch_first=True)
        # out : (batch_size, seq_len, lstm_memory), PackedSequence
        out, (hn, cn) = self.rnn(packed, (h0, c0))
        # out[0] : (batch_size, seq_len, lstm_memory)
        # outputs : (batch_size, seq_len, vocab_size)
        outputs = self.linear2(out[0])
        return outputs

    # feature : (batch_size, lstm_memory)
    # bos : torch.zeros(batch_size) (LongTensor, indices of <BOS>)
    def decode(self, batch_size, bos, image_feature, method='greedy'):
        sentences = [0] * batch_size
        with torch.no_grad():
            # emb_word : (batch_size, emb_size)
            emb_word = self.emb(bos)
            batch_size = image_feature.size(0)
            for i in range(self.max_seqlen):
                # hx : (batch_size, lstm_memory)
                hx = image_feature
                # cx : (batch_size, lstm_memory)
                cx = random.randn(batch_size, self.lstm_memory)
                hx, cx = self.rnn(emb_word, (hx, cx))
                # output : (batch_size, vocab_size)
                output = self.linear(hx)
                # wordidx : (batch_size)
                wordidx = torch.argmax(hx, dim=1)
                emb_word = self.emb(wordidx)
                for bs in batch_size:
                    sentences[bs].append(wordidx[bs].item())
        return sentences

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



