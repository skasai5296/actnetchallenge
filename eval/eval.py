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
from imagemodels.resnet import resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200
from dataset.activitynet_captions import ActivityNetCaptions
import transforms.spatial_transforms as spt
import transforms.temporal_transforms as tpt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/ssd1/dsets/activitynet_captions')
    parser.add_argument('--model_path', type=str, default='../models')
    parser.add_argument('--meta_path', type=str, default='videometa_train.json')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--framepath', type=str, default='frames')
    parser.add_argument('--annpath', type=str, default='train.json')
    parser.add_argument('--cnnmethod', type=str, default='resnet')
    parser.add_argument('--rnnmethod', type=str, default='LSTM')
    parser.add_argument('--vocabpath', type=str, default='vocab.json')
    parser.add_argument('--lstm_pretrain_ep', type=int, default=None)
    parser.add_argument('--model_ep', type=int, default=500)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--lstm_stacks', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--imsize', type=int, default=224)
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--lstm_memory', type=int, default=512)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--max_seqlen', type=int, default=30)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--token_level', action='store_true')
    parser.add_argument('--cuda', action='store_false')
    args = parser.parse_args()


    # gpus
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    print("using {} gpus...".format(n_gpu))

    # load vocabulary
    vocab = Vocabulary(token_level=args.token_level)
    vocpath = os.path.join(args.root_path, args.vocabpath)
    try:
        assert os.path.exists(vocpath)
    except AssertionError:
        print("didn't find vocab in {}! aborting".format(vocpath))
        sys.exit(0)
    vocab.load(vocpath)
    vocab_size = len(vocab)

    # transforms
    sp = spt.Compose([spt.CornerCrop(size=args.imsize), spt.ToTensor()])
    tp = tpt.Compose([tpt.TemporalRandomCrop(args.clip_len), tpt.LoopPadding(args.clip_len)])

    # dataloading
    dset = ActivityNetCaptions(args.root_path, args.meta_path, args.mode, vocab, args.framepath, spatial_transform=sp, temporal_transform=tp)
    dloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, collate_fn=collater, drop_last=True)
    max_it = int(len(dset) / args.batch_size)

    # models
    video_encoder = resnet10(sample_size=args.imsize, sample_duration=args.clip_len)
    caption_gen = RNNCaptioning(method=args.rnnmethod, emb_size=args.embedding_size, lstm_memory=args.lstm_memory, vocab_size=vocab_size, max_seqlen=args.max_seqlen)
    models = [video_encoder, caption_gen]

    # apply pretrained model
    if args.lstm_pretrain_ep is not None:
        dec_model_dir = os.path.join(args.model_path, "{}_pre".format(args.rnnmethod), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
        dec_filename = "ep{:04d}.ckpt".format(args.lstm_pretrain_ep)
        dec_model_path = os.path.join(dec_model_dir, dec_filename)
        try:
            assert os.path.exists(dec_model_path)
        except AssertionError:
            print("didn't find file in directory {}! aborting".format(dec_model_path))
            sys.exit(0)
        caption_gen.load_state_dict(torch.load(dec_model_path, map_location='cuda' if args.cuda and torch.cuda.is_available() else 'cpu'))
        print("loaded pretrained decoder model from {}".format(dec_model_dir))
    else:
        offset = args.model_ep
        if offset > 0:
            enc_model_dir = os.path.join(args.model_path, "{}_{}".format(args.cnnmethod, args.num_layers), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
            enc_filename = "ep{:04d}.ckpt".format(offset)
            enc_model_path = os.path.join(enc_model_dir, enc_filename)
            dec_model_dir = os.path.join(args.model_path, "{}_fine".format(args.rnnmethod), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
            dec_filename = "ep{:04d}.ckpt".format(offset)
            dec_model_path = os.path.join(dec_model_dir, dec_filename)
            try:
                assert os.path.exists(enc_model_path) and os.path.exists(dec_model_path)
            except AssertionError:
                print("didn't find files in directories {} or {}! aborting".format(enc_model_path, dec_model_path))
                sys.exit(0)
            video_encoder.load_state_dict(torch.load(enc_model_path, map_location='cuda' if args.cuda and torch.cuda.is_available() else 'cpu'))
            caption_gen.load_state_dict(torch.load(dec_model_path, map_location='cuda' if args.cuda and torch.cuda.is_available() else 'cpu'))
            print("loaded pretrained models from epoch {}".format(offset))
        else:
            print("cannot set model_ep to number smaller than 1")

    # move models to device
    video_encoder = video_encoder.to(device)
    caption_gen = caption_gen.to(device)

    # count parameters
    num_params = sum(count_parameters(model) for model in models)
    print("# of params in model : {}".format(num_params))

    # eval loop
    print("start evaluation")
    before = time.time()
    for it, data in enumerate(dloader):

        clip, captions, lengths = data

        # move to device
        clip = clip.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        # flow through model
        with torch.no_grad():
            feature = video_encoder(clip)
            feature = feature.view(args.batch_size, args.embedding_size)
            caption = caption_gen.decode(feature, captions)
            caption = vocab.return_sentence(caption)
            for b in range(args.batch_size):
                print(caption[b])

        before = time.time()


    print("end evaluation")





