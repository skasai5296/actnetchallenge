import sys, os
sys.path.append(os.pardir)

import torch
import torch.nn as nn

from imagemodels import resnet, pre_act_resnet, wide_resnet, resnext, densenet
from langmodels.lstm import LSTMCaptioning
from utils.utils import weight_init

def generate_3dcnn(opt):
    assert opt.cnn_name in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    if opt.cnn_name == 'resnet':
        model = resnet.generate_model(
            model_depth=opt.cnn_depth,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.cnn_name == 'wideresnet':
        model = wide_resnet.generate_model(
            model_depth=opt.cnn_depth,
            k=opt.wide_resnet_k,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.cnn_name == 'resnext':
        model = resnext.generate_model(
            model_depth=opt.cnn_depth,
            cardinality=opt.resnext_cardinality,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.cnn_name == 'preresnet':
        model = pre_act_resnet.generate_model(
            model_depth=opt.cnn_depth,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.cnn_name == 'densenet':
        model = densenet.generate_model(
            model_depth=opt.cnn_depth,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)

    if opt.cuda:
        model = model.to(torch.device('cuda'))

    if opt.enc_pretrain_path is not None:
        print('loading pretrained 3DCNN model from {}'.format(opt.enc_pretrain_path))
        pretrain = torch.load(opt.enc_pretrain_path)
        model.load_state_dict(pretrain['state_dict'])
    else:
        model.apply(weight_init)
        print('no path specified, starting 3DCNN model from scratch')


    return model


def generate_rnn(vocab_size, opt):
    assert opt.rnn_name in [
        'LSTM', 'GRU', 'RNN'
    ]

    if opt.rnn_name == 'LSTM':
        model = LSTMCaptioning(ft_size=opt.feature_size, emb_size=opt.embedding_size, lstm_memory=opt.lstm_memory, vocab_size=vocab_size, max_seqlen=opt.max_seqlen, num_layers=opt.rnn_layers, dropout_p=opt.rnn_dropout)
    else:
        raise NotImplementedError

    if opt.cuda:
        model = model.to(torch.device('cuda'))

    if opt.dec_pretrain_path is not None:
        print('loading pretrained RNN model from {}'.format(opt.dec_pretrain_path))
        pretrain = torch.load(opt.dec_pretrain_path)
        model.load_state_dict(pretrain['state_dict'])
    else:
        model.apply(weight_init)
        print('no path specified, starting RNN model from scratch')


    return model
