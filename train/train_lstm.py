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
            print("embedding size not valid for loading.")
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

    """
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
    train_dset = ActivityNetCaptions(args.root_path, args.meta_path, args.mode, vocab, args.framepath, sample_duration=args.clip_len, spatial_transform=sp, temporal_transform=tp)
    trainloader = DataLoader(train_dset, batch_size=args.bs, shuffle=True, num_workers=args.n_cpu, collate_fn=collate_fn, drop_last=True, pin_memory=True)
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
            video_encoder.apply(weight_init)
            caption_gen.apply(weight_init)
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

    # begin training
    begin = time.time()

    # decoder pretraining loop
    if args.lstm_pretrain_ep > 0:
        print("start decoder pretraining, doing for {} epochs".format(args.lstm_pretrain_ep))
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
            dec_save_dir = os.path.join(args.model_save_path, "{}_pre".format(args.langmethod), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
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
        enc_save_dir = os.path.join(args.model_save_path, "{}_{}".format(args.modelname, args.modeldepth), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
        enc_filename = "ep{:04d}.ckpt".format(ep+1)
        enc_save_path = os.path.join(enc_save_dir, enc_filename)
        dec_save_dir = os.path.join(args.model_save_path, "{}_fine".format(args.langmethod), "b{:03d}_s{:03d}_l{:03d}".format(args.bs, args.imsize, args.clip_len))
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
    """





