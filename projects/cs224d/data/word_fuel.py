import numpy as np
import string
import cPickle as pickle
from collections import Counter
from parable.projects.cs224d.ug_cfg import WORD_MAX_SEQ_LEN, WORD_EOS_TOK, WORD_UNK_TOK,\
        SRC_DICT_FILE, TGT_DICT_FILE, SRC_TRAIN_FILE, TGT_TRAIN_FILE,\
        WORD_SORT_K_BATCHES
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.datasets.text import TextFile
from fuel.transformers import (
        Merge, Filter, Batch, SortMapping, Unpack, Padding, Mapping)

class TooLong(object):
    def __init__(self, seq_len):
        self.seq_len = seq_len
    def __call__(self, sents):
        # make sure both shorter than maximum length
        return all([len(sent) <= self.seq_len for sent in sents])

def seq_to_str(seq, bdict):
    return ' '.join([bdict[k] for k in seq])


class SimpleWordLoader:

    '''
    just adapted from SimpleCharLoader in char_fuel.py
    won't work for larger text files

    also assumes that input file has already been procssed with fixed
    vocabulary size and pre-tokenized (with newlines preserved and single-spaces between tokens)
    '''

    def __init__(self, text_file, batch_size, seq_len, train_frac=0.95, valid_frac=0.05):
        if isinstance(text_file, (list, tuple)):
            self.split_init(text_file, batch_size, seq_len, train_frac, valid_frac)
        else:
            self.single_init(text_file, batch_size, seq_len, train_frac, valid_frac)

    def split_init(self, text_file, batch_size, seq_len, train_frac, valid_frac):
        self.batch_size = batch_size
        self.seq_len = seq_len
        assert len(text_file) == 3
        train_file, valid_file, test_file = text_file

        with open(train_file, 'r') as fin:
            text = fin.read()
        text = text.split(' ')
        counts = Counter(text)
        self._counts = counts
        #self._counts[' '] = 0  # TODO can also try perserving spaces
        total_count = sum(self._counts.values())
        self.norm_freq = dict((k, v/float(total_count)) for k, v in self._counts.iteritems())
        vocab = dict()
        for idx, val in enumerate(counts):
            vocab[val] = idx

        with open(valid_file, 'r') as fin:
            valid_text = fin.read().split(' ')
        with open(test_file, 'r') as fin:
            test_text = fin.read().split(' ')

        remainder = len(text) % (batch_size * seq_len)
        # FIXME hack
        text = text[:-remainder]
        assert(len(text) % (batch_size * seq_len) == 0)
        remainder = len(valid_text) % (batch_size * seq_len)
        valid_text = valid_text[:-remainder]
        remainder = len(test_text) % (batch_size * seq_len)
        test_text = test_text[:-remainder]

        self.splits = dict()
        inds = [vocab[c] for c in text]
        self.splits['train'] = np.array(inds).reshape((batch_size, -1)).astype(np.int32)
        valid_inds = [vocab[c] for c in valid_text]
        self.splits['val']  = np.array(valid_inds).reshape((batch_size, -1)).astype(np.int32)
        test_inds = [vocab[c] for c in test_text]
        self.splits['test'] = np.array(test_inds).reshape((batch_size, -1)).astype(np.int32)

        self.vocab = vocab

        l = seq_len
        self.num_train_batches = self.splits['train'].shape[1] / l
        self.num_val_batches = self.splits['val'].shape[1] / l
        self.num_test_batches = self.splits['test'].shape[1] / l

    # split up single file ourselves
    def single_init(self, text_file, batch_size, seq_len, train_frac, valid_frac):
        self.batch_size = batch_size
        self.seq_len = seq_len

        with open(text_file, 'r') as fin:
            text = fin.read()
        text = text.split(' ')
        counts = Counter(text)
        self._counts = counts
        #self._counts[' '] = 0  # TODO can also try perserving spaces
        total_count = sum(self._counts.values())
        self.norm_freq = dict((k, v/float(total_count)) for k, v in self._counts.iteritems())
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

def load_dictionaries():
    with open(SRC_DICT_FILE, 'rb') as fin:
        src_dict = pickle.load(fin)
    with open(TGT_DICT_FILE, 'rb') as fin:
        tgt_dict = pickle.load(fin)
    return src_dict, tgt_dict

# also refer to machine_translation/stream.py
def load_data(src_file, tgt_file, batch_size, sort_k_batches, training=False):
    src_dict, tgt_dict = load_dictionaries()

    src_dset = TextFile(files=[src_file], dictionary=src_dict,
            bos_token=None, eos_token=None, unk_token=WORD_UNK_TOK)
    tgt_dset = TextFile(files=[tgt_file], dictionary=tgt_dict,
            bos_token=WORD_EOS_TOK, eos_token=WORD_EOS_TOK, unk_token=WORD_UNK_TOK)

    stream = Merge([src_dset.get_example_stream(), tgt_dset.get_example_stream()],
            ('source', 'target'))
    # filter sequences that are too long
    if training:
        stream = Filter(stream, predicate=TooLong(seq_len=WORD_MAX_SEQ_LEN))
        # batch and read k batches ahead
        stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size*sort_k_batches))
        # sort all samples in read-ahead batch
        stream = Mapping(stream, SortMapping(lambda x: len(x[1])))
        # turn back into stream
        stream = Unpack(stream)
    # batch again
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))
    # NOTE pads with zeros so eos_idx should be 0
    masked_stream = Padding(stream)
    return masked_stream, src_dict, tgt_dict

if __name__ == '__main__':
    # test things out
    src_dict, tgt_dict = load_dictionaries()
    print(len(src_dict))
    # note we don't use <s> so can subtract 1 from here (</s> and <s> have same index)
    print(len(src_dict))

    batch_size = 4
    train_stream, src_dict, tgt_dict = load_data(SRC_TRAIN_FILE, TGT_TRAIN_FILE, batch_size, WORD_SORT_K_BATCHES, training=True)

    batches = 4
    batch_ind = 0
    for ss, sm, ts, tm in train_stream.get_epoch_iterator():
        print(ss)
        print(sm)
        print(ts)
        print(tm)
        print 'lengths:', np.sum(tm, axis=1)
        batch_ind += 1
        if batch_ind >= batches:
            break
