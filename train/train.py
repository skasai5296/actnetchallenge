import sys, os
import time
import shutil
import argparse

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
from langmodels.bert import Bert
from langmodels.vocab import Vocabulary
import imagemodels.resnet as resnet
from dataset.activitynet_captions import ActivityNetCaptions
import transforms.spatial_transforms as spt
import transforms.temporal_transforms as tpt

from options import parse_args
from utils.makemodel import generate_model

if __name__ == '__main__':
    args = parse_args()

    print(args)

    # gpus
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # load vocabulary
    vocab = Vocabulary(token_level=args.token_level)
    vocpath = os.path.join(args.root_path, args.vocabpath)
    if not os.path.exists(vocpath):
        vocab.add_corpus(os.path.join(args.root_path, args.annpath))
        vocab.save(vocpath)
    else:
        vocab.load(vocpath)
    vocab_size = len(vocab)

    # transforms
    sp = spt.Compose([spt.CornerCrop(size=args.imsize), spt.ToTensor()])
    tp = tpt.Compose([tpt.TemporalRandomCrop(args.clip_len), tpt.LoopPadding(args.clip_len)])

    # dataloading
    train_dset = ActivityNetCaptions(args.root_path, args.meta_path, args.mode, vocab, args.framepath, spatial_transform=sp, temporal_transform=tp)
    trainloader = DataLoader(train_dset, batch_size=args.bs, shuffle=True, num_workers=args.n_cpu, collate_fn=collater, drop_last=True)
    max_it = int(len(train_dset) / args.bs)

    # models
    video_encoder = generate_model(args)
    caption_gen = RNNCaptioning(method=args.rnnmethod, emb_size=args.embedding_size, ft_size=args.feature_size, lstm_memory=args.lstm_memory, vocab_size=vocab_size, max_seqlen=args.max_seqlen, num_layers=args.lstm_stacks)
    models = [video_encoder, caption_gen]

    # apply pretrained model
    offset = args.start_from_ep
    if offset != 0:
        enc_model_dir = os.path.join(args.model_path, "{}_{}".format(args.modelname, args.modeldepth), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
        enc_filename = "ep{:04d}.ckpt".format(offset)
        enc_model_path = os.path.join(enc_model_dir, enc_filename)
        dec_model_dir = os.path.join(args.model_path, "{}_fine".format(args.rnnmethod, args.lstm_stacks), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
        dec_filename = "ep{:04d}.ckpt".format(offset)
        dec_model_path = os.path.join(dec_model_dir, dec_filename)
        if os.path.exists(enc_model_path) and os.path.exists(dec_model_path):
            video_encoder.load_state_dict(torch.load(enc_model_path))
            caption_gen.load_state_dict(torch.load(dec_model_path))
            print("restarting training from epoch {}".format(offset))
        else:
            offset = 0
            video_encoder.apply(weight_init)
            caption_gen.apply(weight_init)
            print("didn't find file, starting encoder, decoder from scratch")

    # initialize pretrained embeddings
    if args.emb_init is not None:
        lookup = get_pretrained_from_txt(args.emb_init)
        assert len(list(lookup.values())[0]) == args.embedding_size
        matrix = torch.randn_like(caption_gen.emb.weight)
        for char, vec in lookup.items():
            if char in vocab.obj2idx.keys():
                id = vocab.obj2idx[char]
                matrix[id, :] = torch.tensor(vec)
        caption_gen.init_embedding(matrix)
        print("succesfully initialized embeddings from {}".format(args.emb_init))

    # move models to device
    video_encoder = video_encoder.to(device)
    caption_gen = caption_gen.to(device)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1 and args.dataparallel:
        video_encoder = nn.DataParallel(video_encoder)
        caption_gen = nn.DataParallel(caption_gen)
    else:
        n_gpu = 1
    print("using {} gpus...".format(n_gpu))

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer, scheduler
    params = list(video_encoder.parameters()) + list(caption_gen.parameters())
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience, verbose=True)

    # count parameters
    num_params = sum(count_parameters(model) for model in models)
    print("# of params in model : {}".format(num_params))

    assert args.max_epochs > offset, "already at offset epoch number, aborting training"

    # decoder pretraining loop
    print("start decoder pretraining, doing for {} epochs".format(args.lstm_pretrain_ep))
    before = time.time()
    for ep in range(args.lstm_pretrain_ep):
        for it, data in enumerate(trainloader):

            _, captions, lengths = data

            optimizer.zero_grad()

            # move to device
            captions = captions.to(device)
            lengths = torch.tensor([min(args.max_seqlen, length) for length in lengths], dtype=torch.long, device=device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # flow through model, but if pretraining lstm, use a 0 vector for video features
            feature = torch.zeros(args.bs, args.feature_size).to(device)
            caption, length = caption_gen(feature, captions, lengths)

            # lengths returned by caption_gen should be distributed because of dataparallel, so merge.
            centered = []
            for gpu in range(n_gpu):
                centered.extend([ten[gpu].item() for ten in length])
            packedcaption = pack_padded_sequence(caption, centered, batch_first=True)[0]

            # backpropagate loss and store negative log likelihood
            nll = criterion(packedcaption, targets)
            nll.backward()
            optimizer.step()

            # log losses
            if it % args.log_every == (args.log_every-1):
                after = time.time()
                print("iter {:06d}/{:06d} | nll loss: {:.04f} | {:02.04f}s per loop".format(it+1, max_it, nll.cpu().item(), (after-before)/args.log_every), flush=True)
                before = time.time()

        scheduler.step(nll.cpu().item())
        print("epoch {:04d}/{:04d} done (pretrain), loss: {:.06f}".format(ep+1, args.lstm_pretrain_ep, nll.cpu().item()), flush=True)

        # save models
        dec_save_dir = os.path.join(args.model_path, "{}_pre".format(args.rnnmethod), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
        dec_filename = "ep{:04d}.ckpt".format(ep+1)
        dec_save_path = os.path.join(dec_save_dir, dec_filename)
        if not os.path.exists(dec_save_dir):
            os.makedirs(dec_save_dir)

        if n_gpu > 1 and args.dataparallel:
            torch.save(caption_gen.module.state_dict(), dec_save_path)
        else:
            torch.save(caption_gen.state_dict(), dec_save_path)
        print("saved pretrained decoder model to {}".format(dec_save_path))
        scheduler.step(nll.cpu().item())


    # joint training loop
    print("start training")
    before = time.time()
    for ep in range(offset, args.max_epochs):
        for it, data in enumerate(trainloader):

            clip, captions, lengths = data

            optimizer.zero_grad()

            # move to device
            clip = clip.to(device)
            captions = captions.to(device)
            lengths = torch.tensor([min(args.max_seqlen, length) for length in lengths], dtype=torch.long, device=device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # flow through model, but if pretraining lstm, use a 0 vector for video features
            feature = video_encoder(clip)
            feature = feature.view(args.bs, args.feature_size)
            caption, length = caption_gen(feature, captions, lengths)

            # lengths returned by caption_gen should be distributed because of dataparallel, so merge.
            centered = []
            for gpu in range(n_gpu):
                centered.extend([ten[gpu].item() for ten in length])
            packedcaption = pack_padded_sequence(caption, centered, batch_first=True)[0]

            # backpropagate loss and store negative log likelihood
            nll = criterion(packedcaption, targets)
            nll.backward()
            optimizer.step()

            # log losses
            if it % args.log_every == (args.log_every-1):
                after = time.time()
                print("iter {:06d}/{:06d} | nll loss: {:.04f} | {:02.04f}s per loop".format(it+1, max_it, nll.cpu().item(), (after-before)/args.log_every), flush=True)
                before = time.time()

        scheduler.step(nll.cpu().item())
        print("epoch {:04d}/{:04d} done, loss: {:.06f}".format(ep+1, args.max_epochs, nll.cpu().item()), flush=True)

        # save models
        enc_save_dir = os.path.join(args.model_path, "{}_{}".format(args.modelname, args.modeldepth), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
        enc_filename = "ep{:04d}.ckpt".format(ep+1)
        enc_save_path = os.path.join(enc_save_dir, enc_filename)
        dec_save_dir = os.path.join(args.model_path, "{}_fine".format(args.rnnmethod), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
        dec_filename = "ep{:04d}.ckpt".format(ep+1)
        dec_save_path = os.path.join(dec_save_dir, dec_filename)
        if not os.path.exists(enc_save_dir):
            os.makedirs(enc_save_dir)
        if not os.path.exists(dec_save_dir):
            os.makedirs(dec_save_dir)

        if n_gpu > 1 and args.dataparallel:
            torch.save(video_encoder.module.state_dict(), enc_save_path)
            torch.save(caption_gen.module.state_dict(), dec_save_path)
        else:
            torch.save(video_encoder.state_dict(), enc_save_path)
            torch.save(caption_gen.state_dict(), dec_save_path)
        print("saved encoder model to {}".format(enc_save_path))
        print("saved decoder model to {}".format(dec_save_path))


        before = time.time()


    print("end training")





