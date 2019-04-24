import sys, os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

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
                 emb_size=512,
                 lstm_memory=512,
                 vocab_size=100,
                 max_seqlen=20
                 ):
        super(RNNCaptioning, self).__init__()
        self.method = method
        self.emb_size = emb_size
        self.lstm_memory = lstm_memory
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen

        self.emb = nn.Embedding(self.vocab_size, self.emb_size)
        if method == 'LSTM':
            self.rnn = nn.LSTMCell(self.emb_size, self.lstm_memory)
        self.linear = nn.Linear(self.lstm_memory, self.vocab_size)
        self.softmax = nn.Softmax()

    # THWC must be flattened for image feature
    # image_feature : (batch_size, lstm_memory)
    # x : (batch_size, seq_len) (LongTensor of indexes)
    def forward(self, image_feature, x):
        # emb_seq : (batch_size, seq_len, emb_size)
        emb_seq = self.emb(x)
        batch_size = image_feature.size(0)
        seq_len = x.size(1)
        # emb_seq : (seq_len, batch_size, emb_size)
        emb_seq = emb_seq.transpose(1, 0)
        # hx : (batch_size, lstm_memory)
        hx = image_feature
        # cx : (batch_size, lstm_memory)
        cx = torch.randn_like(hx)
        output = []
        # xthemb : (seq_len)
        # (hx, cx) : (batch_size, vocab_size)
        for xthemb in emb_seq:
            hx, cx = self.rnn(xthemb, (hx, cx))
            output.append(hx)
        # output : (batch_size, seq_len, lstm_memory)
        output = torch.stack(output, dim=0).transpose(0, 1)
        # output : (batch_size, seq_len, vocab_size)
        output = self.softmax(self.linear(output))
        return output

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



