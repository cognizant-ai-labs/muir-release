#
# Functions for loading Wikitext-2 dataset.
#
# Adapted from https://github.com/pytorch/examples/blob/master/word_language_model/
#

import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        print("Train")
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        print("Val")
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        print("Test")
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        print("Tokens", tokens)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class LanguageModelingDataset(object):

    def __init__(self, batchified_data, bptt):
        self.source = batchified_data
        self.bptt = bptt
        self.current = 0
        print("Dataset Length:", len(self))

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self):
            raise StopIteration
        batch = get_batch(self.source, self.bptt, self.current * self.bptt)
        self.current += 1
        return batch

    def next(self):
        return self.__next__()

    def __len__(self):
        return self.source.size(0) // self.bptt - 1


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, bptt, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def load_wikitext2(batch_size=20):

    data_dir = os.path.expanduser('~/hyperdatasets') + '/wikitext-2'

    print("Creating Corpus")
    corpus = Corpus(data_dir)

    print("Batchifying")
    eval_batch_size = 10
    print("Train")
    train_data = batchify(corpus.train, batch_size)
    print("Val")
    val_data = batchify(corpus.valid, eval_batch_size)
    print("Test")
    test_data = batchify(corpus.test, eval_batch_size)

    print("Creating Datasets")
    bptt = 35
    trainloader = LanguageModelingDataset(train_data, bptt)
    valloader = LanguageModelingDataset(val_data, bptt)
    testloader = LanguageModelingDataset(test_data, bptt)

    classes = None

    return trainloader, valloader, testloader, classes


if __name__ == '__main__':
    trainloader, valloader, testloader, classes = load_wikitext2()
    train_iter = iter(trainloader)
    for i in range(2):
        print(train_iter.next())

