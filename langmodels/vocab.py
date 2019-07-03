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

# return indexes of sentence batch as torch.LongTensor
def return_idx(sentence_batch, text_proc):
    out = []
    for sentence in sentence_batch:
        processed = parse(sentence)
        idx = []
        for token in processed:
            idx.append(sent_proc.stoi(token))
        out.append(idx)
    return torch.tensor(out, dtype=torch.long)

# return sentence batch from indexes from torch.LongTensor
def return_sentences(ten, text_proc):
    ten = ten.tolist()
    out = []
    for idxs in ten:
        tokenlist = [text_proc.itos[idx] for idx in idxs]
        out.append(" ".join(tokenlist))
    return out

# for debugging
if __name__ == '__main__':

    text_proc = build_vocab(["/ssd1/dsets/activitynet_captions/train.json", "/ssd1/dsets/activitynet_captions/val_1.json", "/ssd1/dsets/activitynet_captions/val_2.json"])
    print(parse("the cat and the hat sat on a mat", text_proc))

