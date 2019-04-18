import sys, os
import time


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
def tokenize(sentence):
    tokens = ['<BOS>']
    for token in sentence.split():
        cleaned = clean_word(token)
        if not is_number(cleaned):
            tokens.append(cleaned)
        else:
            tokens.append('<NUM>')
    tokens.append('<EOS>')
    return tokens


# Vocabulary class. set cshow to True for character-level training
class Vocabulary():
    def __init__(self, cshow=False):
        self.idx2obj = []
        self.obj2idx = {}
        self.idx2obj.append('<BOS>')
        self.idx2obj.append('<EOS>')
        self.idx2obj.append('<UNK>')
        self.idx2obj.append('<NUM>')
        self.set_obj2idx()
        self.len = 4
        self.cshow = cshow

    def set_obj2idx(self):
        for idx, obj in enumerate(self.idx2obj):
            self.obj2idx[obj] = idx

    def __len__(self):
        return self.len

    # add text file including corpus to add to the dictionary.
    """
    def add_corpus(self, txtpath):
        before = time.time()
        with open(txtpath, 'r') as f:
            for line in f:
                if not self.cshow:
                    tokens = line.split()
                    for token in tokens:
                        cleaned = clean_word(tokens)
                        self.add_token(cleaned)
                else:
                    tokens = line
                    for token in tokens:
                        self.add_token(token)
        print("added corpus to dictionary, {:.4f}s taken".format(time.time() - before), flush=True)
    """

    # add strings (sentences or single tokens, not chars) to dictionary
    def add_sentence(self, sentence):
        for token in tokenize(sentence):
            if token not in self.idx2obj:
                self.idx2obj.append(token)
                self.obj2idx[token] = self.len
                self.len += 1

    # write ordered list of words in vocab to json file in path
    def write(self, path):
        with open(path, "a+") as f:
            f.write(json.dumps(self.idx2obj))
        print("wrote dictionary to {}".format(path), flush=True)

    # return list of indices for given sentence
    def return_idx(self, sentence):
        idxs = []
        if not self.cshow:
            for cleaned in tokenize(sentence):
                try:
                    idxs.append(self.obj2idx[cleaned])
                except KeyError:
                    idxs.append(self.obj2idx['<UNK>'])
        else:
            for char in sentence:
                idxs.append(self.obj2idx[char])
        return idxs

    # TODO: add pretrained weights.
    def add_pretrained(self, jsonpath):
        pass


# for debugging
if __name__ == '__main__':
    sentence = "The cat and the rat on a very small mat. They put on 2 hats and sat on the mat!"
    vocab = Vocabulary()
    vocab.add_sentence(sentence)
    print(vocab.idx2obj)
    print(vocab.return_idx(sentence))

