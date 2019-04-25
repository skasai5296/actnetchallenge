import sys, os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim

sys.path.append(os.pardir)
from utils.utils import *
from langmodels.lstm import RNNCaptioning
from langmodels.vocab import Vocabulary
from imagemodels.resnet import resnet10
from dataset.activitynet_captions import ActivityNetCaptions
import transforms.spatial_transforms as spt
import transforms.temporal_transforms as tpt

if __name__ == '__main__':

    # params
    root_path = "../../../ssd1/dsets/activitynet_captions"
    model_path = "../models"
    meta_path = 'videometa_train.json'
    mode = 'train'
    framepath = 'frames'
    prepath = 'train.json'
    imsize = 224
    clip_len = 16
    bs = 64
    n_cpu = 8
    lstm_memory = 512
    embedding_size = 512
    max_seqlen = 30
    max_epochs = 20
    lr=1e-2
    momentum=0.9
    lstm_stacks = 3
    token_level = False
    vocabpath = "vocab.json"
    cuda = True
    dataparallel = True
    rnnmethod = 'LSTM'


    # gpus
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    print("using {} gpus...".format(n_gpu))

    # load vocabulary
    vocab = Vocabulary(token_level=token_level)
    vocpath = os.path.join(root_path, vocabpath)
    if not os.path.exists(vocpath):
        vocab.add_corpus(os.path.join(root_path, prepath))
        vocab.save(vocpath)
    else:
        vocab.load(vocpath)
    vocab_size = len(vocab)

    # transforms
    sp = spt.Compose([spt.CornerCrop(size=imsize), spt.ToTensor()])
    tp = tpt.Compose([tpt.TemporalRandomCrop(clip_len), tpt.LoopPadding(clip_len)])

    # dataloading
    train_dset = ActivityNetCaptions(root_path, meta_path, mode, vocab, framepath, spatial_transform=sp, temporal_transform=tp)
    trainloader = DataLoader(train_dset, batch_size=bs, shuffle=True, num_workers=n_cpu, collate_fn=collater, drop_last=True)
    max_it = int(len(train_dset) / bs)

    # models
    video_encoder = resnet10(sample_size=imsize, sample_duration=clip_len)
    caption_gen = RNNCaptioning(method=rnnmethod, emb_size=embedding_size, lstm_memory=lstm_memory, vocab_size=vocab_size, max_seqlen=max_seqlen, num_layers=lstm_stacks)
    models = [video_encoder, caption_gen]

    # move models to device
    video_encoder = video_encoder.to(device)
    caption_gen = caption_gen.to(device)
    if n_gpu > 1 and dataparallel:
        video_encoder = nn.DataParallel(video_encoder)
        caption_gen = nn.DataParallel(caption_gen)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    params = list(video_encoder.parameters()) + list(caption_gen.parameters())
    optimizer = optim.SGD(params, lr=lr, momentum=momentum)

    # count parameters
    num_params = sum(count_parameters(model) for model in models)
    print("# of params in model : {}".format(num_params))

    # training loop
    for ep in range(max_epochs):
        for it, data in enumerate(trainloader):

            clip, captions, lengths = data

            optimizer.zero_grad()

            # move to device
            clip = clip.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)

            feature = video_encoder(clip)
            feature = feature.view(bs, embedding_size)
            caption = caption_gen(feature, captions, lengths)

            tmp = torch.zeros(bs, 1, dtype=torch.long).to(device)
            targets = torch.cat((captions[:, 1:], tmp), dim=1)

            targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]
            nll = criterion(caption, targets)
            nll.backward()
            optimizer.step()

            if it % 10 == 9:
                print("iter {:06d}/{:06d} | nll loss: {:.04f}".format(it+1, max_it, nll.cpu().item()), flush=True)
        print("epoch {:04d}/{:04d} done, loss: {:.06f}".format(ep+1, max_epochs, nll.cpu().item()), flush=True)

        enc_save_path = os.path.join(model_path, "{}_{}".format("resnet", "10"), "ep{}".format(ep+1))
        dec_save_path = os.path.join(model_path, "{}_{}".format(rnnmethod, lstm_stacks), "ep{}".format(ep+1))
        torch.save(video_encoder.state_dict(), enc_save_path)
        torch.save(caption_gen.state_dict(), dec_save_path)





