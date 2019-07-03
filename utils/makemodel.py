import sys, os
sys.path.append(os.pardir)

import torch
import torch.nn as nn

from imagemodels import resnet, pre_act_resnet, wide_resnet, resnext, densenet
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

    if not opt.pretrain_path:
        model.apply(weight_init)
        print('no path specified, starting 3DCNN model from scratch')
        return model, model.parameters()

    print('loading pretrained 3DCNN model from {}'.format(opt.pretrain_path))
    pretrain = torch.load(opt.pretrain_path)
    assert opt.arch == pretrain['arch']

    model.load_state_dict(pretrain['state_dict'])

    parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)

    return model, parameters

def generate_lstm(opt):
    assert opt.rnn_name in [
        'LSTM', 'GRU', 'RNN'
    ]

    model = getattr(nn, opt.rnn_name)(input_size=, hidden_size=, num_layers, batch_first=True, dropout=opt.rnn_dropout, bidirectional=False)
    if opt.rnn_name == 'res:
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

    if not opt.pretrain_path:
        model.apply(weight_init)
        print('no path specified, starting 3DCNN model from scratch')
        return model, model.parameters()

    print('loading pretrained 3DCNN model from {}'.format(opt.pretrain_path))
    pretrain = torch.load(opt.pretrain_path)
    assert opt.arch == pretrain['arch']

    model.load_state_dict(pretrain['state_dict'])

    parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)

    return model, parameters
