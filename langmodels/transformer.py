import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax()

    # q, k : (bs x seq x d_k)
    # v : (bs x seq x d_v)
    # mask : (bs x seq x seq)
    def forward(self, q, k, v, mask=None):

        # attn : (bs x seq x seq)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn, dim=2)
        attn = self.dropout(attn)
        # output : (bs x seq x d_v)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # heads are created independently from original q, k, v using linear layers
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    # q, k, v : (bs x seq x d_model)
    # mask : (bs x seq x seq)
    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # q, k : (bs x seq x n_head x d_k)
        # q, k : (bs x seq x n_head x d_v)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # q, k : (n_head*bs x seq x d_k)
        # v : (n_head*bs x seq x d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        # mask : (n_head*bs x seq x seq)
        mask = mask.repeat(n_head, 1, 1)
        # output : (n_head*bs x seq x d_v)
        # attn : (n_head*bs x seq x seq)
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        # output : (bs x seq x n_head*d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        # output : (bs x seq x d_model)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    # x : (bs x seq x d_model)
    def forward(self, x):
        residual = x
        # output : (bs x d_model x seq)
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        # output : (bs x seq x d_model)
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    # enc_input : (bs x seq x d_model)
    # non_pad_mask : (bs x seq x d_model), mask the padded part of sequence
    # slf_attn_mask : (bs x seq x seq), mask the subsequent tokens in encoder
    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        # enc_slf_attn : (n_head*bs x seq x seq)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # masked multiheadattention
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # multiheadattention with q, k from encoder
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    # dec_input, enc_output : (bs x seq x d_model)
    # non_pad_mask : (bs x seq x d_model), mask the padded part of sequence
    # slf_attn_mask : (bs x seq x seq), mask the subsequent tokens in decoder
    # dec_enc_attn_mask : (bs x d_k x d_k), mask for giving from encoder to decoder
    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


