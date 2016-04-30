import numpy as np
import string
from collections import Counter
from parable.projects.cs224d.ug_cfg import CHAR_MAX_SEQ_LEN, CHAR_EOS_TOK, CHAR_SOS_TOK, CHAR_UNK_TOK
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.datasets.text import TextFile
from fuel.transformers import (
        Merge, Filter, Batch, SortMapping, Unpack, Padding, Mapping)

class TooLong(object):
    def __init__(self, seq_len):
        self.seq_len = seq_len
    def __call__(self, sents):
        # target sentence is artificially longer due to eos
        return len(sents[0]) <= self.seq_len

def seq_to_str(seq, bdict):
    return ''.join([bdict[k] for k in seq])

# just single text for language modeling, reshape into matrix style

class SimpleCharLoader:

    def __init__(self, text_file, batch_size, seq_len, train_frac=0.95, valid_frac=0.05):
        self.batch_size = batch_size
        self.seq_len = seq_len

        with open(text_file, 'r') as fin:
            text = fin.read()
        counts = Counter(text)
        vocab = dict()
        for idx, val in enumerate(counts):
            vocab[val] = idx

        # remove extra
        remainder = len(text) % (batch_size * seq_len)
        text = text[:-remainder]
        assert(len(text) % (batch_size * seq_len) == 0)
        inds = [vocab[c] for c in text]
        inds = np.array(inds).reshape((batch_size, -1)).astype(np.int32)

        self.inds = inds
        self.vocab = vocab

        total_len = inds.shape[1]
        test_frac = 1.0 - train_frac - valid_frac
        assert(test_frac >= 0.0)
        val_len = int(total_len * valid_frac)
        test_len = int(total_len * test_frac)
        train_len = total_len - val_len - test_len

        self.vocab = vocab
        self.splits = dict()
        self.splits['train'] = inds[:, 0:train_len]
        self.splits['val'] = inds[:, train_len:train_len+val_len]
        self.splits['test'] = inds[:, train_len+val_len:]

        l = seq_len
        self.num_train_batches = train_len / l
        self.num_val_batches = val_len / l
        self.num_test_batches = test_len / l

    def get_batch(self, split, batch_ind):
        l = self.seq_len
        x = self.splits[split][:, batch_ind * l:(batch_ind + 1) * l]
        y = self.splits[split][:, batch_ind * l + 1: (batch_ind+1) * l + 1]
        if y.shape[1] != x.shape[1]:
            z = self.splits[split][:, 0][:, None]
            y = np.concatenate([z, y], axis=1)
        return x, y
    def shuffle(self, split):
        print 'Shuffling...'
        self.splits[split] = self.splits[split].T
        np.random.shuffle(self.splits[split])
        self.splits[split] = self.splits[split].T

from collections import Counter

class ShufflingCharLoader:

    def __init__(self, text_file, batch_size, seq_len, train_frac=0.95, valid_frac=0.05):
        self.batch_size = batch_size
        self.seq_len = seq_len

        with open(text_file, 'r') as fin:
            text = fin.read()
        counts = Counter(text)
        vocab = dict()
        for idx, val in enumerate(counts):
            vocab[val] = idx

        # remove extra

        remainder = len(text) % (batch_size * seq_len)
        text = text[:-remainder]
        assert(len(text) % (batch_size * seq_len) == 0)
        inds = [vocab[c] for c in text]
        inds = np.array(inds).reshape((len(text)/seq_len, seq_len)).astype(np.int32)

        self.vocab = vocab

        total_len = inds.shape[0]

        test_frac = 1.0 - train_frac - valid_frac
        assert(test_frac >= 0.0)
        val_len = int(total_len * valid_frac)
        test_len = int(total_len * test_frac)
        train_len = total_len - val_len - test_len

        self.vocab = vocab
        self.splits = dict()
        self.splits['train'] = inds[:train_len, :]
        self.splits['val'] = inds[train_len:train_len+val_len, :]
        self.splits['test'] = inds[train_len+val_len:, :]

        l = seq_len
        self.num_train_batches = train_len / l
        self.num_val_batches = val_len / l
        self.num_test_batches = test_len / l

    def get_batch(self, split, batch_ind):
        l = self.batch_size
        x = self.splits[split][batch_ind * l:(batch_ind + 1) * l, :-1]
        y = self.splits[split][batch_ind * l: (batch_ind+1) * l, 1:]

        return x, y
    def shuffle(self, split):
        print 'Shuffling...'
        np.random.shuffle(self.splits[split])

# also refer to machine_translation/stream.py
def load_parallel_data(src_file, tgt_file, batch_size, sort_k_batches, dictionary, training=False):
    def preproc(s):
        s = s.replace('``', '"')
        s = s.replace('\'\'', '"')
        return s
    enc_dset = TextFile(files=[src_file], dictionary=dictionary,
            bos_token=None, eos_token=None, unk_token=CHAR_UNK_TOK, level='character', preprocess=preproc)
    dec_dset = TextFile(files=[tgt_file], dictionary=dictionary,
            bos_token=CHAR_SOS_TOK, eos_token=CHAR_EOS_TOK, unk_token=CHAR_UNK_TOK, level='character', preprocess=preproc)
    # NOTE merge encoder and decoder setup together
    stream = Merge([enc_dset.get_example_stream(), dec_dset.get_example_stream()],
            ('source', 'target'))
    if training:
        # filter sequences that are too long
        stream = Filter(stream, predicate=TooLong(seq_len=CHAR_MAX_SEQ_LEN))
        # batch and read k batches ahead
        stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size*sort_k_batches))
        # sort all samples in read-ahead batch
        stream = Mapping(stream, SortMapping(lambda x: len(x[1])))
        # turn back into stream
        stream = Unpack(stream)
    # batch again
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))
    masked_stream = Padding(stream)
    return masked_stream
