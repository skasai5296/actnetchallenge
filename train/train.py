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
from utils.utils import *
from langmodels.lstm import RNNCaptioning
from langmodels.bert import Bert
from langmodels.transformer import Transformer
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
    torch.backends.cudnn.benchmark=True

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
    collatefn = functools.partial(collater, args.max_seqlen)
    train_dset = ActivityNetCaptions(args.root_path, args.meta_path, args.mode, vocab, args.framepath, sample_duration=args.clip_len, spatial_transform=sp, temporal_transform=tp)
    trainloader = DataLoader(train_dset, batch_size=args.bs, shuffle=True, num_workers=args.n_cpu, collate_fn=collatefn, drop_last=True, pin_memory=True)
    max_it = int(len(train_dset) / args.bs)

    # models
    video_encoder = generate_model(args)
    # rewrite part of average pooling
    if args.langmethod == 'Transformer':
        scale = 16
        inter_time = int(args.clip_len/scale)
        video_encoder.avgpool = nn.AdaptiveAvgPool3d((inter_time, 1, 1))
        video_encoder.fc = Identity()
    if args.langmethod == 'LSTM':
        caption_gen = RNNCaptioning(method=args.langmethod, emb_size=args.embedding_size, ft_size=args.feature_size, lstm_memory=args.lstm_memory, vocab_size=vocab_size, max_seqlen=args.max_seqlen, num_layers=args.lstm_stacks)
    elif args.langmethod == 'Transformer':
        caption_gen = Transformer(d_ft=args.feature_size, n_tgt_vocab=vocab_size, len_max_seq=args.max_seqlen, d_word_vec=args.embedding_size, d_model=args.lstm_memory, d_inner=2048, n_layers=6,
                                    n_head=8, d_k=64, d_v=64, dropout=0.1, tgt_emb_prj_weight_sharing=True)
    models = [video_encoder, caption_gen]


    # apply pretrained model
    offset = args.start_from_ep
    if offset != 0:
        enc_model_dir = os.path.join(args.model_path, "{}_{}".format(args.modelname, args.modeldepth), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
        enc_filename = "ep{:04d}.ckpt".format(offset)
        enc_model_path = os.path.join(enc_model_dir, enc_filename)
        dec_model_dir = os.path.join(args.model_path, "{}_fine".format(args.langmethod, args.lstm_stacks), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
        dec_filename = "ep{:04d}.ckpt".format(offset)
        dec_model_path = os.path.join(dec_model_dir, dec_filename)
        if os.path.exists(enc_model_path) and os.path.exists(dec_model_path):
            video_encoder.load_state_dict(torch.load(enc_model_path))
            caption_gen.load_state_dict(torch.load(dec_model_path))
            print("restarting training from epoch {}".format(offset))
        else:
            offset = 0
            """
            video_encoder.apply(weight_init)
            caption_gen.apply(weight_init)
            """
            print("didn't find file, starting encoder, decoder from scratch")


    # initialize pretrained embeddings
    if args.langmethod == 'LSTM' and args.emb_init is not None:
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
    if args.langmethod == 'LSTM':
        criterion = nn.CrossEntropyLoss()
    elif args.langmethod == 'Transformer':
        criterion = LabelSmoothingLoss(label_smoothing=0.1, out_classes=len(vocab))

    # optimizer, scheduler
    params = list(video_encoder.parameters()) + list(caption_gen.parameters())
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience, verbose=True)

    # count parameters
    num_params = sum(count_parameters(model) for model in models)
    print("# of params in model : {}".format(num_params))

    assert args.max_epochs > offset, "already at offset epoch number, aborting training"

    # decoder pretraining loop
    print("start decoder pretraining, doing for {} epochs".format(args.lstm_pretrain_ep))
    begin = time.time()
    before = time.time()
    for ep in range(args.lstm_pretrain_ep):
        for it, data in enumerate(trainloader):

            _, captions, lengths, _, _ = data

            optimizer.zero_grad()

            # move to device
            captions = captions.to(device)
            if args.langmethod == 'LSTM':
                lengths = torch.tensor([min(args.max_seqlen, length) for length in lengths], dtype=torch.long, device=device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            else:
                targets = captions

            # flow through model, use a 0 vector for video features
            if args.langmethod == 'Transformer':
                feature = torch.zeros(args.bs, args.max_seqlen, args.feature_size).to(device)
                pos = torch.arange(args.max_seqlen).repeat(args.batchsize, 1).to(device) + 1
                for b, length in enumerate(lengths):
                    pos[b, length:] = 0
                caption = caption_gen(feature, captions, pos, targets, pos)
            else:
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
                print("{}, iter {:06d}/{:06d} | nll loss: {:.04f} | {:02.04f}s per loop".format(sec2str(time.time()-begin), it+1, max_it, nll.cpu().item(), (after-before)/args.log_every), flush=True)
                before = time.time()

        print("{}, epoch {:04d}/{:04d} done (pretrain), loss: {:.06f}".format(sec2str(time.time()-begin), ep+1, args.lstm_pretrain_ep, nll.cpu().item()), flush=True)

        # save models
        dec_save_dir = os.path.join(args.model_path, "{}_pre".format(args.langmethod), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
        dec_filename = "ep{:04d}.ckpt".format(ep+1)
        dec_save_path = os.path.join(dec_save_dir, dec_filename)
        if not os.path.exists(dec_save_dir):
            os.makedirs(dec_save_dir)

        if n_gpu > 1 and args.dataparallel:
            torch.save(caption_gen.module.state_dict(), dec_save_path)
        else:
            torch.save(caption_gen.state_dict(), dec_save_path)
        print("saved pretrained decoder model to {}".format(dec_save_path))
        # scheduler.step(nll.cpu().item())
    print("done with decoder pretraining")

    # joint training loop
    print("start training")
    before = time.time()
    for ep in range(offset, args.max_epochs):
        for it, data in enumerate(trainloader):

            clip, captions, lengths, _, _ = data

            optimizer.zero_grad()

            # move to device
            clip = clip.to(device)
            captions = captions.to(device)
            if args.langmethod == 'LSTM':
                lengths = torch.tensor([min(args.max_seqlen, length) for length in lengths], dtype=torch.long, device=device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            elif args.langmethod == 'Transformer':
                targets = captions

            # flow through model (no backprop to CNN model)
            with torch.no_grad():
                feature = video_encoder(clip)
            if args.langmethod == 'Transformer':
                # feature : (bs x C' x T/16 x 1 x 1) -> (bs x T/16 x C')
                # faster maybe?
                feature = feature.squeeze(-1).squeeze(-1).transpose(1, 2)
                # feature = feature.permute(0, 2, 1, 3, 4).contiguous().view(args.bs, inter_time, args.feature_size)
                # feature : (bs x T/16 x C'H'W')

                if args.max_seqlen <= inter_time:
                    pad_feature = feature[:, :args.max_seqlen, :]
                else:
                    pad_feature = torch.zeros(args.bs, args.max_seqlen, args.feature_size).to(device)
                    pad_feature[:, :inter_time, :] = feature

                # positional encodings
                src_pos = torch.arange(args.max_seqlen).repeat(args.bs, 1).to(device) + 1
                tgt_pos = torch.arange(args.max_seqlen).repeat(args.bs, 1).to(device) + 1
                caption = caption_gen(pad_feature, src_pos, captions, tgt_pos)
            elif args.langmethod == 'LSTM':
                # feature : (bs x C' x 1 x 1 x 1)
                feature = feature.squeeze(-1).squeeze(-1).squeeze(-1)
                caption = caption_gen(feature, captions, lengths)
                """
                # lengths returned by caption_gen should be distributed because of dataparallel, so merge.
                centered = []
                for gpu in range(n_gpu):
                    centered.extend([ten[gpu].item() for ten in length])
                """

            # backpropagate loss and store negative log likelihood
            nll = criterion(caption, targets)
            nll.backward()
            # gradient norm clipping
            torch.nn.utils.clip_grad_norm_(caption_gen.parameters(), max_norm=1.0)
            optimizer.step()

            # log losses
            if it % args.log_every == (args.log_every-1):
                after = time.time()
                print("{} | iter {:06d}/{:06d} | nll loss: {:.04f} | {:02.04f}s per loop".format(sec2str(time.time()-begin), it+1, max_it, nll.cpu().item(), (after-before)/args.log_every), flush=True)
                before = time.time()

        # scheduler.step(nll.cpu().item())
        print("{}, epoch {:04d}/{:04d} done, loss: {:.06f}".format(sec2str(time.time()-begin), ep+1, args.max_epochs, nll.cpu().item()))
        if args.langmethod == 'Transformer':
            print("sample sentences:")
            for sentence in vocab.return_sentence(caption.argmax(dim=-1)):
                print(sentence, flush=True)


        # save models
        enc_save_dir = os.path.join(args.model_path, "{}_{}".format(args.modelname, args.modeldepth), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
        enc_filename = "ep{:04d}.ckpt".format(ep+1)
        enc_save_path = os.path.join(enc_save_dir, enc_filename)
        dec_save_dir = os.path.join(args.model_path, "{}_fine".format(args.langmethod), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
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





