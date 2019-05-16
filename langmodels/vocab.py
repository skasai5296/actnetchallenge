import sys, os
import time
import json


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# strip unneeded characters and make lowercase
def clean_word(token):
    cleaned = token.strip("!@#$%^&*()_+|~{}:\"\'<>?-=\\`[];',./").lower()
    return cleaned

# clean words in sentence and return list with <BOS>, <EOS> and <NUM> included
def tokenize(sentence, token_level=False):
    tokens = ['<BOS>']
    if not token_level:
        for token in sentence.split():
            cleaned = clean_word(token)
            if not is_number(cleaned):
                tokens.append(cleaned)
            else:
                tokens.append('<NUM>')
    else:
        tokens.extend([*sentence])
    tokens.append('<EOS>')
    return tokens


# Vocabulary class. set token_level to True for character-level training
class Vocabulary():
    def __init__(self, token_level=False):
        self.idx2obj = []
        self.obj2idx = {}
        self.idx2obj.append('<PAD>')
        self.idx2obj.append('<BOS>')
        self.idx2obj.append('<EOS>')
        self.idx2obj.append('<UNK>')
        if not token_level:
            self.idx2obj.append('<NUM>')
        self.len = len(self.idx2obj)
        self.set_obj2idx()
        self.token_level = token_level

    def set_obj2idx(self):
        for idx, obj in enumerate(self.idx2obj):
            self.obj2idx[obj] = idx

    # add vocab from json file including corpus.
    # DESIGNED SPECIFICALLY FOR ACTIVITYNET CAPTIONS
    def add_corpus(self, path):
        before = time.time()
        with open(path, 'r') as f:
            obj = json.load(f)
            for _, data in obj.items():
                captions = data['sentences']
                for sentence in captions:
                    self.add_sentence(sentence)
        print("added corpus to dictionary, {:.4f}s taken".format(time.time() - before), flush=True)

    # add strings (sentences or single tokens, not chars) to dictionary
    def add_sentence(self, sentence):
        for token in tokenize(sentence, token_level=self.token_level):
            if token not in self.idx2obj:
                self.idx2obj.append(token)
                self.obj2idx[token] = self.len
                self.len += 1

    # write ordered list of words in vocab to json file in path
    def save(self, path):
        if os.path.exists(path):
            print("dictionary already exists in {}, did not save", flush=True)
        else:
            with open(path, "a+") as f:
                json.dump(self.idx2obj, f)
            print("wrote dictionary to {}".format(path), flush=True)

    # get vocab from json file in path
    def load(self, path):
        with open(path, "r") as f:
            self.idx2obj = json.load(f)
        self.set_obj2idx()
        print("loaded dictionary from {}".format(path), flush=True)
        print("dictionary length: {} words".format(len(self.idx2obj)))

    # return list of indices for given sentence
    def return_idx(self, sentence):
        idxs = []
        if not self.token_level:
            for cleaned in tokenize(sentence, token_level=self.token_level):
                try:
                    idxs.append(self.obj2idx[cleaned])
                except KeyError:
                    idxs.append(self.obj2idx['<UNK>'])
        else:
            for char in sentence.lstrip(' '):
                idxs.append(self.obj2idx[char])
        return idxs

    # return sentence from given index list
    # idxlist : torch.Tensor, (bs, *)
    def return_sentence(self, idxlist):
        idxlist = idxlist.cpu().tolist()
        sentences = []
        for idlist in idxlist:
            sentence = []
            for tokenid in idlist:
                token = self.idx2obj[tokenid]
                sentence.append(token)
                if token == '<EOS>':
                    break
            if not self.token_level:
                sentence = " ".join(sentence)
            else:
                sentence = "".join(sentence)
            sentences.append(sentence)

        return sentences

    def _return_sentence(self, idxlist):
        sentence = []
        for tokenid in idxlist:
            token = self.idx2obj[tokenid]
            sentence.append(token)
            if token == '<EOS>':
                break
        if not self.token_level:
            sentence = " ".join(sentence)
        else:
            sentence = "".join(sentence)

        return sentence


    def __len__(self):
        return len(self.idx2obj)


# for debugging
if __name__ == '__main__':
    sentence = "The cat and the rat on a very small mat. They put on 2 hats and sat on the mat!"
    vocab = Vocabulary(token_level=False)
    vocab.add_sentence(sentence)
    print(vocab.idx2obj)
    idxs = vocab.return_idx(sentence)
    print(idxs)
    cap = vocab._return_sentence(idxs)
    print(cap)

