import sys, os
import time
import shutil
import argparse
import functools

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim

sys.path.append(os.pardir)
from utils.utils import sec2str, count_parameters, weight_init, get_pretrained_from_txt
from langmodels.lstm import LSTMCaptioning
from langmodels.vocab import build_vocab, return_idx, return_sentences
from dataset.activitynet_train import ActivityNetCaptions_Train
from dataset.activitynet_valtest import ActivityNetCaptions_Val
import transforms.spatial_transforms as spt
import transforms.temporal_transforms as tpt

from options import parse_args
from utils.makemodel import generate_3dcnn, generate_rnn

def train_lstm(args):
    # gpus
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # load vocabulary
    annfiles = [os.path.join(args.root_path, pth) for pth in args.annpaths]
    text_proc = build_vocab(annfiles, args.min_freq, args.max_seqlen)
    vocab_size = len(text_proc.vocab)

    # transforms
    sp = spt.Compose([spt.CornerCrop(size=args.imsize), spt.ToTensor()])
    tp = tpt.Compose([tpt.TemporalRandomCrop(args.clip_len), tpt.LoopPadding(args.clip_len)])

    # dataloading
    train_dset = ActivityNetCaptions_Train(args.root_path, ann_path='train_fps.json', n_samples_for_each_video=1, sample_duration=args.clip_len, spatial_transform=sp, temporal_transform=tp)
    trainloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)
    max_train_it = int(len(train_dset) / args.batch_size)
    val_dset = ActivityNetCaptions_Val(args.root_path, ann_path=['val_1_fps.json', 'val_2_fps.json'], n_samples_for_each_video=1, sample_duration=args.clip_len, spatial_transform=sp, temporal_transform=tp)
    valloader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)
    max_val_it = int(len(val_dset) / args.batch_size)

    # models
    video_encoder = generate_3dcnn(args)
    caption_gen = generate_rnn(vocab_size, args)
    models = [video_encoder, caption_gen]

    # initialize pretrained embeddings
    if args.emb_init is not None:
        begin = time.time()
        print("initializing embeddings from {}...".format(args.emb_init))
        lookup = get_pretrained_from_txt(args.emb_init)
        first = next(iter(lookup.values()))
        try:
            assert len(first) == args.embedding_size
        except AssertionError:
            print("embedding size not compatible with pretrained embeddings.")
            print("specified size {}, pretrained model includes size {}".format(args.embedding_size, len(first)))
            sys.exit(1)
        matrix = torch.randn_like(caption_gen.emb.weight)
        for char, vec in lookup.items():
            if char in text_proc.vocab.stoi.keys():
                id = text_proc.vocab.stoi[char]
                matrix[id, :] = torch.tensor(vec)
        caption_gen.init_embedding(matrix)
        print("{} | successfully initialized".format(sec2str(time.time() - begin), args.emb_init))

    # move models to device
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1 and args.dataparallel:
        video_encoder = nn.DataParallel(video_encoder)
        caption_gen = nn.DataParallel(caption_gen)
    else:
        n_gpu = 1
    print("using {} gpus...".format(n_gpu))

    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index=text_proc.vocab.stoi['<pad>'])

    # optimizer, scheduler
    params = list(video_encoder.parameters()) + list(caption_gen.parameters())
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=args.patience, verbose=True)

    # count parameters
    num_params = sum(count_parameters(model) for model in models)
    print("# of params in model : {}".format(num_params))

    # joint training loop
    print("start training")
    begin = time.time()
    for ep in range(args.max_epochs):

        # train for epoch
        video_encoder, caption_gen, optimizer = train_epoch(trainloader, video_encoder, caption_gen, optimizer, criterion, device, text_proc, log_interval=args.log_every, max_it=max_train_it)

        # save models
        enc_save_dir = os.path.join(args.model_save_path, "encoder")
        enc_filename = "ep{:04d}.pth".format(ep+1)
        if not os.path.exists(enc_save_dir):
            os.makedirs(enc_save_dir)
        enc_save_path = os.path.join(enc_save_dir, enc_filename)

        dec_save_dir = os.path.join(args.model_save_path, "decoder")
        dec_filename = "ep{:04d}.pth".format(ep+1)
        dec_save_path = os.path.join(dec_save_dir, dec_filename)
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

        # evaluate
        print("begin evaluation for epoch {} ...".format(ep+1))
        nll, ppl, sample = validate(valloader, video_encoder, caption_gen, criterion, device, text_proc, log_interval=args.log_every, max_it=max_val_it)
        scheduler.step(ppl)

        print("{}, epoch {:04d}/{:04d} done, validation loss: {:.06f}, perplexity: {:.03f}".format(sec2str(time.time()-begin), ep+1, args.max_epochs, nll, ppl))

    print("end training")


def train_epoch(trainloader, encoder, decoder, optimizer, criterion, device, text_proc, log_interval, max_it):

    ep_begin = time.time()
    before = time.time()
    for it, data in enumerate(trainloader):
        # TODO: currently supports only batch size of 1, enable more in the future
        ids = data['id'][0]
        durations = data['duration'][0]
        sentences = data['sentences'][0]
        timestamps = data['timestamps'][0]
        fps = data['fps'][0]
        clip = data['clips'][0]

        captions = return_idx(sentences, text_proc)

        optimizer.zero_grad()

        # move to device
        clip = clip.to(device)
        captions = captions.to(device)
        target = captions.clone().detach()

        # flow through model
        feature = encoder(clip)
        # feature : (bs x C')

        output = decoder(feature, captions)
        # caption : (batch_size, vocab_size, seq_len)

        # backpropagate loss and store negative log likelihood
        nll = criterion(output, target)
        nll.backward()
        # gradient norm clipping
        # torch.nn.utils.clip_grad_norm_(caption_gen.parameters(), max_norm=1.0)
        optimizer.step()

        # log losses
        if it % log_interval == (log_interval-1):
            print("epoch {} | iter {:06d}/{:06d} | nll loss: {:.04f} | {:02.04f}s per loop".format(sec2str(time.time()-ep_begin), it+1, max_it, nll.cpu().item(), (time.time()-before)/log_interval), flush=True)
            before = time.time()

    return encoder, decoder, optimizer

def validate(valloader, encoder, decoder, criterion, device, text_proc, log_interval, max_it):
    encoder.eval()
    decoder.eval()
    nll_list = []
    ppl_list = []
    begin = time.time()
    before = time.time()
    with torch.no_grad():
        for it, data in enumerate(valloader):
            # TODO: currently supports only batch size of 1, enable more in the future
            ids = data['id'][0]
            durations = data['duration'][0]
            sentences = data['sentences'][0]
            timestamps = data['timestamps'][0]
            fps = data['fps'][0]
            clip = data['clips'][0]

            captions = return_idx(sentences, text_proc)

            # move to device
            clip = clip.to(device)
            captions = captions.to(device)
            target = captions.clone().detach()

            # flow through model
            # feature : (bs x C')
            feature = encoder(clip)

            # output : (batch_size, vocab_size, seq_len)
            try:
                output = decoder.sample(feature, captions)
            # workaround for dataparallel
            except AttributeError:
                output = decoder.module.sample(feature, captions)

            # sample : (seq_len)
            sample = output.max(1)

            # backpropagate loss and store negative log likelihood
            nll = criterion(output, target).cpu().item()
            nll_list.append(nll)
            ppl = 2 ** nll
            ppl_list.append(ppl)

            if it % log_interval == (log_interval-1):
                print("validation {} | iter {:06d}/{:06d} | perplexity: {:.04f} | {:02.04f}s per loop".format(sec2str(time.time()-begin), it+1, max_it, sum(ppl_list)/len(ppl_list), (time.time()-before)/log_interval), flush=True)
                before = time.time()
                samplesentence = return_sentences(sample, text_proc)
                print("sample sentences: ")
                print(samplesentence)

            # evaluate for only 100 iterations
            if it % 100 == 99:
                break

    meannll = sum(nll_list) / len(nll_list)
    meanppl = sum(ppl_list) / len(ppl_list)

    return meannll, meanppl


if __name__ == '__main__':
    args = parse_args()

    print(args)

    train_lstm(args)






