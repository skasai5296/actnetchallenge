import sys, os
sys.path.append(os.pardir)
import time
import json

import torch
import torchtext
import spacy

from utils.utils import sec2str

sp = spacy.load('en_core_web_sm')

# builds vocabulary from annotation files (can include validation) using spacy tokenizer
def build_vocab(jsonfiles, min_freq=5, max_len=30):
    before = time.time()
    print("building vocabulary...", flush=True)
    text_proc = torchtext.data.Field(sequential=True, init_token="<bos>", eos_token="<eos>", lower=True, fix_length=max_len, tokenize="spacy", batch_first=True)
    sentences = []
    for jsonfile in jsonfiles:
        with open(jsonfile, 'r') as f:
            alldata = json.load(f)
        for obj in alldata.values():
            for sentence in obj['sentences']:
                sentences.append(sentence.strip())
    sent_proc = list(map(text_proc.preprocess, sentences))
    text_proc.build_vocab(sent_proc, min_freq=min_freq)
    print("done building vocabulary, minimum frequency is {} times".format(min_freq))
    print("{} | # of words in vocab: {}".format(sec2str(time.time() - before), len(text_proc.vocab)))
    return text_proc

# parse sentence
def parse(sentence, text_proc):
    processed = text_proc.preprocess(sentence.strip())
    return processed

# sentence_batch: list of str
# return indexes of sentence batch as torch.LongTensor
def return_idx(sentence_batch, text_proc):
    out = []
    preprocessed = list(map(text_proc.preprocess, sentence_batch))
    out = text_proc.process(preprocessed)
    return out

# return sentence batch from indexes from torch.LongTensor
def return_sentences(ten, text_proc):
    if isinstance(ten, torch.Tensor):
        ten = ten.tolist()
    out = []
    for idxs in ten:
        tokenlist = [text_proc.vocab.itos[idx] for idx in idxs]
        out.append(" ".join(tokenlist))
    return out

# for debugging
if __name__ == '__main__':

    files = ["train.json, val_1.json, val_2.json"]
    files = [os.path.join("/ssd1/dsets/activitynet_captions", pth) for pth in files]
    text_proc = build_vocab(files)
    sentence = ["the cat and the hat sat on a mat"]
    ten = return_idx(sentence, text_proc)
    print(ten)
    sent = return_sentences(ten, text_proc)
    print(sent)

