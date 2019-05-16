import sys, os

import numpy as np

sys.path.append(os.pardir)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.init as init
import torch.nn.functional as F

from langmodels.vocab import tokenize


PAD = 0

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    Calculate the cross entropy loss.
    """
    def __init__(self, label_smoothing, out_classes, ignore_index=PAD):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()
        self.ignore_index = ignore_index
        self.neg_value = label_smoothing / (out_classes - 1)
        self.pos_value = 1.0 - label_smoothing
        self.out_classes = out_classes

    def forward(self, output, target):
        """
        output (FloatTensor): (bs x seqlen x C), logits of classes
        target (LongTensor): (bs x seqlen), output class indexes
        """
        one_hot = torch.full_like(output, self.neg_value).scatter(-1, target.unsqueeze(-1), self.pos_value)
        log_probs = F.log_softmax(output, dim=-1)
        non_pad_mask = target.ne(self.ignore_index)
        cematrix = -(one_hot * log_probs).sum(dim=-1)
        loss = cematrix.masked_select(non_pad_mask).mean()

        return loss

def collater(maxlen, datas, token_level=False):
    # sort data by caption lengths for packing
    datas.sort(key=lambda x: x[1].size(0), reverse=True)
    clips, captions = zip(*datas)
    batchsize = len(captions)
    ten = []
    lengths = torch.tensor([cap.size(0) for cap in captions], dtype=torch.long)
    padded_captions = torch.zeros(batchsize, maxlen, dtype=torch.long)
    for i, caption in enumerate(captions):
        length = min(caption.size(0), maxlen)
        padded_captions[i, :length] = caption[:length]

    batched_clip = torch.stack(clips)
    return batched_clip, padded_captions, lengths

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_pretrained_from_txt(path):
    with open(path, "r") as f:
        lookup = {}
        for line in f:
            emb = line.rstrip().split()
            token = emb[0]
            vec = [float(num) for num in emb[1:]]
            lookup[token] = vec
    return lookup


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


# returns the mask containing 1s for not padding indexes
# seq : (bs, seq_len)
def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)

# returns sinusoid encoding for given position and dimension.
# torch.FloatTensor(n_position x d_hid)
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

# returns the mask for padding part of sequence
def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

# returns the mask for only taking account of precedent tokens
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


