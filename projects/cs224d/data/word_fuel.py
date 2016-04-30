import numpy as np
import string
import cPickle as pickle
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
