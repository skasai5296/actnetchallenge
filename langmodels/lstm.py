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
        super(LSTMCaptioning, self).__init__()
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
    # returns : (batch_size, vocab_size, seq_len)
    def forward(self, image_feature, captions, init_state=None):
        # feature : (batch_size, 1, emb_size)
        feature = self.linear1(image_feature).unsqueeze(1)
        # captions : (batch_size, seq_len, emb_size)
        captions = self.emb(captions)
        # inseq : (batch_size, 1+seq_len, emb_size)
        inseq = torch.cat((feature, captions), dim=1)
        # hiddens : (batch_size, seq_len, lstm_memory)
        hiddens, _ = self.rnn(inseq[:, :-1, :])
        # outputs : (batch_size, vocab_size, seq_len)
        outputs = self.linear2(hiddens).transpose(1, 2)
        return outputs

    # image_feature : (batch_size, ft_size)
    # method : one of ['greedy', 'beamsearch']
    def sample(self, image_feature, method='greedy', init_state=None):
        outputlist = []
        # inputs : (batch_size, 1, emb_size)
        inputs = self.linear1(image_feature).unsqueeze(1)
        states = init_state
        for idx in range(self.max_seqlen):
            hiddens, states = self.rnn(inputs, states)
            # outputs : (batch_size, vocab_size)
            outputs = self.linear2(hiddens.squeeze(1))
            outputlist.append(outputs)
            # pred : (batch_size), LongTensor
            _, pred = outputs.max(1)
            # inputs : (batch_size, emb_size)
            inputs = self.emb(pred)
            inputs = inputs.unsqueeze(1)
        # sampled : (batch_size, vocab_size, max_seqlen)
        sampled = torch.stack(outputlist, 2)
        return sampled

    # TODO
    def init_pretrained_weights(self, vocab):
        self.emb.init_pretrained_weights(vocab)


class Attention(nn.Module):
    def __init__(self,
                 encoder_dim,
                 decoder_dim,
                 attention_dim
                 ):
        super(Attention, self).__init__()
        self.input_enc = nn.Linear(encoder_dim, attention_dim)
        self.output_enc = nn.Linear(decoder_dim, attention_dim)
        self.att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    # feature : (batch_size, T*H*W, encoder_dim)
    # dec_out : (batch_size, decoder_dim)
    def forward(self, feature, dec_out):
        out1 = self.input_enc(feature)
        out2 = self.output_enc(dec_out)
        # out : (batch_size, T*H*W, attention_dim)
        out = self.relu(out1, out2.unsqueeze(1))
        # att : (batch_size, T*H*W)
        att = self.att(out).unsqueeze(-1)
        # alpha : (batch_size, T*H*W)
        alpha = self.softmax(att)
        # weighted : (batch_size, encoder_dim)
        weighted = (feature * alpha.unsqueeze(-1)).sum(dim=1)
        return alpha, weighted


class LSTMCaptioning_Attention(nn.Module):
    def __init__(self,
                 enc_dim=512,
                 lstm_memory=512,
                 attention_size=512,
                 emb_size=512,
                 vocab_size=100,
                 max_seqlen=20,
                 num_layers=1,
                 dropout_p=0.1
                 ):
        super(LSTMCaptioning, self).__init__()
        self.enc_dim = enc_dim
        self.lstm_memory = lstm_memory
        self.attention_size = attention_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen
        self.num_layers = num_layers

        self.attention = Attention(enc_dim, lstm_memory, attention_size)
        self.init_h = nn.Linear(enc_dim, lstm_memory)
        self.init_c = nn.Linear(enc_dim, lstm_memory)
        self.gate = nn.Linear(enc_dim, lstm_memory)
        self.emb = nn.Embedding(vocab_size, emb_size)
        #self.rnn = nn.LSTM(self.emb_size, self.lstm_memory, self.num_layers, batch_first=True, dropout=dropout_p)
        self.rnn = nn.LSTMCell(emb_size+enc_dim, lstm_memory)
        self.sigmoid = nn.Sigmoid()
        self.outlinear = nn.linear(lstm_memory, vocab_size)

    def init_embedding(self, tensor):
        self.emb.weight.data.copy_(tensor)

    # THWC must be flattened for image feature
    # image_feature : (batch_size, T, H, W, enc_dim)
    # captions : (batch_size, seq_len)
    # returns : (batch_size, vocab_size, seq_len)
    def forward(self, image_feature, captions):
        bs = image_feature.size(0)
        dim = image_feature.size(-1)
        assert dim == self.enc_dim
        # feature : (batch_size, T*H*W, enc_dim)
        feature = image_feature.view(bs, -1, dim)
        pixnum = feature.size(1)

        # mean_ft : (batch_size, enc_dim)
        mean_ft = feature.mean(dim=1)
        # hn, cn : (batch_size, lstm_memory)
        hn = self.init_h(mean_ft)
        cn = self.init_c(mean_ft)

        outputlist = []
        # inputs : (batch_size, 1, emb_size)
        for idx in range(self.max_seqlen):
            weighted, alpha = self.attention(feature, hn)
            hiddens, cn = self.rnn(inputs, cn)
            # outputs : (batch_size, vocab_size)
            outputs = self.linear2(hiddens.squeeze(1))
            outputlist.append(outputs)
            # pred : (batch_size), LongTensor
            _, pred = outputs.max(1)
            # inputs : (batch_size, emb_size)
            inputs = self.emb(pred)
            inputs = inputs.unsqueeze(1)
        # sampled : (batch_size, vocab_size, max_seqlen)
        sampled = torch.stack(outputlist, 2)


        # captions : (batch_size, seq_len, emb_size)
        captions = self.emb(captions)
        # inseq : (batch_size, 1+seq_len, emb_size)
        inseq = torch.cat((feature, captions), dim=1)
        # hiddens : (batch_size, seq_len, lstm_memory)
        hiddens, _ = self.rnn(inseq[:, :-1, :])
        # outputs : (batch_size, vocab_size, seq_len)
        outputs = self.linear2(hiddens).transpose(1, 2)
        return outputs

    # image_feature : (batch_size, ft_size)
    # method : one of ['greedy', 'beamsearch']
    def sample(self, image_feature, method='greedy', init_state=None):
        outputlist = []
        # inputs : (batch_size, 1, emb_size)
        inputs = self.linear1(image_feature).unsqueeze(1)
        states = init_state
        for idx in range(self.max_seqlen):
            hiddens, states = self.rnn(inputs, states)
            # outputs : (batch_size, vocab_size)
            outputs = self.linear2(hiddens.squeeze(1))
            outputlist.append(outputs)
            # pred : (batch_size), LongTensor
            _, pred = outputs.max(1)
            # inputs : (batch_size, emb_size)
            inputs = self.emb(pred)
            inputs = inputs.unsqueeze(1)
        # sampled : (batch_size, vocab_size, max_seqlen)
        sampled = torch.stack(outputlist, 2)
        return sampled


if __name__ == '__main__':
    bs = 10
    data = torch.randn(bs, 512)
    captionidx = torch.randint(0, 100, size=(bs, 20))
    model = RNNCaptioning()
    print(model(data, captionidx).size())



