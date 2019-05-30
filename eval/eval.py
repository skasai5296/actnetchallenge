import sys, os
import time
import shutil
import argparse
import functools
import json

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
    dset = ActivityNetCaptions(args.root_path, args.meta_path, args.mode, vocab, args.framepath, spatial_transform=sp, temporal_transform=tp, sample_duration=args.clip_len)
    dloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, collate_fn=collatefn, drop_last=True)
    max_it = int(len(dset) / args.batch_size)

    # models
    video_encoder, params = generate_model(args)
    # rewrite part of average pooling
    if args.langmethod == 'Transformer':
        scale = 16
        inter_time = int(args.clip_len/scale)
        video_encoder.avgpool = nn.AdaptiveAvgPool3d((inter_time, 1, 1))
        video_encoder.fc = Identity()

    if args.langmethod == 'LSTM':
        caption_gen = RNNCaptioning(method=args.langmethod, emb_size=args.embedding_size, ft_size=args.feature_size,
                lstm_memory=args.lstm_memory, vocab_size=vocab_size, max_seqlen=args.max_seqlen, num_layers=args.lstm_stacks)
    elif args.langmethod == 'Transformer':
        caption_gen = Transformer(n_tgt_vocab=vocab_size, len_max_seq=args.max_seqlen, d_ft=args.feature_size, 
                d_word_vec=args.embedding_size, d_model=args.lstm_memory, d_inner=2048,
                n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, tgt_emb_prj_weight_sharing=True)
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
            print(enc_model_path)
            video_encoder.load_state_dict(torch.load(enc_model_path, map_location='cuda' if args.cuda and torch.cuda.is_available() else 'cpu'))
            caption_gen.load_state_dict(torch.load(dec_model_path, map_location='cuda' if args.cuda and torch.cuda.is_available() else 'cpu'))
            print("loaded trained models from epoch {}".format(offset))
        else:
            print("cannot set model_ep to number smaller than 1")

    # gpus
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()

    # move models to device
    video_encoder = video_encoder.to(device)
    caption_gen = caption_gen.to(device)

    """
    if n_gpu > 1 and args.dataparallel:
        video_encoder = nn.DataParallel(video_encoder)
        caption_gen = nn.DataParallel(caption_gen)
    else:
        n_gpu = 1
    print("using {} gpus...".format(n_gpu))
    """

    # count parameters
    num_params = sum(count_parameters(model) for model in models)
    print("# of params in model : {}".format(num_params))

    # eval loop
    print("start evaluation")
    before = time.time()

    if args.json_path is not None:
        print("reading from a previously made json file at {}".format(args.json_path), flush=True)
        with open(args.json_path, "r") as f:
            submission = json.load(f)
            obj = submission["results"]
    else:
        print("making json file from scratch", flush=True)
        submission = {"version": "VERSION 1.3", "external_data": {"used": True, "details": "Excluding the last fc layer, the video encoding model (3D-ResneXt-101) is pre-trained on the Kinetics-400 training set"}}
        obj = {}

    max_loop = 1

    for loop in range(max_loop):
        for it, data in enumerate(dloader):

            if args.mode not in ['test', 'val']:
                clip, captions, lengths, ids, regs = data
                lengths = torch.tensor([min(args.max_seqlen, length) for length in lengths], dtype=torch.long, device=device)
            else:
                clip, _, _, ids, regs = data
                captions = torch.ones(args.batch_size, args.max_seqlen, dtype=torch.long, device=device)
                lengths = None

            # move to device
            clip = clip.to(device)
            if args.mode not in ['test', 'val']:
                captions = captions.to(device)

            # flow through model
            with torch.no_grad():
                feature = video_encoder(clip)

                if args.langmethod == 'Transformer':
                    feature = feature.squeeze(-1).squeeze(-1).transpose(1, 2)
                    if args.max_seqlen <= inter_time:
                        pad_feature = feature[:, :args.max_seqlen, :]
                    else:
                        pad_feature = torch.zeros(args.batch_size, args.max_seqlen, args.feature_size).to(device)
                        pad_feature[:, :inter_time, :] = feature

                    # positional encodings
                    src_pos = torch.arange(args.max_seqlen).repeat(args.batch_size, 1).to(device) + 1
                    tgt_pos = torch.arange(args.max_seqlen).repeat(args.batch_size, 1).to(device) + 1
                    caption = caption_gen.sample(pad_feature, src_pos, tgt_pos, args.max_seqlen)

                    for c, id, reg in zip(caption, ids, regs):
                        cap = vocab.return_sentence(c.unsqueeze(0))[0]
                        if "v_" + id not in obj.keys():
                            obj["v_" + id] = []
                        obj["v_" + id].append({"sentence": cap, "timestamp": reg})
                        print("id: {} {} {}".format(id, reg, cap), flush=True)

                elif args.langmethod == 'LSTM':
                    caption, length = caption_gen(feature, captions, lengths)
                    # lengths returned by caption_gen should be distributed because of dataparallel, so merge.
                    centered = []
                    for gpu in range(n_gpu):
                        centered.extend([ten[gpu].item() for ten in length])

                    caption = caption_gen.decode(feature, captions)
                    for c, id, reg in zip(caption, ids, regs):
                        cap = vocab.return_sentence(c.unsqueeze(0))[0]
                        if "v_" + id not in obj.keys():
                            obj["v_" + id] = []
                        obj["v_" + id].append({"sentence": cap, "timestamp": reg})
                        #print("id:{} | {} | {}".format(id, reg, cap), flush=True)

            if it % 10 == 9:
                print("progress: {}/{}".format(it+1, max_it))

            before = time.time()
        print("-"*100)
        print("{}/{} loops done".format(loop+1, max_loop))
        print("-"*100)

    submission["results"] = obj

    with open(args.submission_path, "w") as f:
        json.dump(submission, f)
    print("end creation of json file, saved at {}".format(args.submission_path))

    print("evaluation done")





