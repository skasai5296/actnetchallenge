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
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim

sys.path.append(os.pardir)
from utils.utils import *
from langmodels.lstm import RNNCaptioning
from langmodels.bert import Bert
from langmodels.transformer import Transformer
from langmodels.vocab import Vocabulary
from imagemodels.resnet import resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200
from dataset.activitynet_captions import ActivityNetCaptions
import transforms.spatial_transforms as spt
import transforms.temporal_transforms as tpt
from utils.makemodel import generate_model

from options import parse_args

if __name__ == '__main__':

    args = parse_args()


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
    collatefn = functools.partial(collater, args.max_seqlen)
    dset = ActivityNetCaptions(args.root_path, args.meta_path, args.mode, vocab, args.framepath, spatial_transform=sp, temporal_transform=tp)
    dloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, collate_fn=collatefn, drop_last=True)
    max_it = int(len(dset) / args.batch_size)

    # models
    video_encoder, params = generate_model(args)
    if args.langmethod == 'LSTM':
        caption_gen = RNNCaptioning(method=args.langmethod, emb_size=args.embedding_size, ft_size=args.feature_size,
                lstm_memory=args.lstm_memory, vocab_size=vocab_size, max_seqlen=args.max_seqlen, num_layers=args.lstm_stacks)
    elif args.langmethod == 'Transformer':
        caption_gen = Transformer(ftsize=args.feature_size, n_src_vocab=vocab_size, n_tgt_vocab=vocab_size,
                len_max_seq=args.max_seqlen, d_word_vec=args.embedding_size, d_model=args.lstm_memory, d_inner=2048,
                n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, tgt_emb_prj_weight_sharing=True, 
                emb_src_tgt_weight_sharing=True)
    models = [video_encoder, caption_gen]

    # apply pretrained model
    if args.lstm_pretrain_ep is not None:
        dec_model_dir = os.path.join(args.model_path, "{}_pre".format(args.langmethod), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
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
            enc_model_dir = os.path.join(args.model_path, "{}_{}".format(args.modelname, args.modeldepth), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
            enc_filename = "ep{:04d}.ckpt".format(offset)
            enc_model_path = os.path.join(enc_model_dir, enc_filename)
            dec_model_dir = os.path.join(args.model_path, "{}_fine".format(args.langmethod), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
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

        if args.mode is not 'test':
            clip, captions, lengths = data
            lengths = torch.tensor([min(args.max_seqlen, length) for length in lengths], dtype=torch.long, device=device)
        else:
            clip, _, _ = data
            captions = torch.zeros(args.batch_size, args.max_seqlen)
            lengths = None

        # move to device
        clip = clip.to(device)
        if args.mode is not 'test':
            captions = captions.to(device)

        # flow through model
        with torch.no_grad():
            feature = video_encoder(clip)
            feature = feature.view(args.batch_size, args.feature_size)

            if args.langmethod == 'Transformer':
                # positional encodings
                pos = torch.arange(args.max_seqlen).repeat(args.batch_size, 1).to(device) + 1
                if args.mode is not 'test':
                    for b, length in enumerate(lengths):
                        pos[b, length:] = 0
                caption = caption_gen.sample(feature, captions, pos, captions, pos)
                for cap in caption:
                    out = cap.argmax(dim=-1)
                    out = out.unsqueeze(0) if out.dim() == 1 else out
                    c = vocab.return_sentence(out)
                    print(c)


            elif args.langmethod == 'LSTM':
                caption, length = caption_gen(feature, captions, lengths)
                # lengths returned by caption_gen should be distributed because of dataparallel, so merge.
                centered = []
                for gpu in range(n_gpu):
                    centered.extend([ten[gpu].item() for ten in length])

                caption = caption_gen.decode(feature, captions)
                caption = vocab.return_sentence(caption)
                for c in caption:
                    print(c)

        before = time.time()


    print("end evaluation")





