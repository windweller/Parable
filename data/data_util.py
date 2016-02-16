import numpy as np
import theano
import cPickle as pkl


def padding_single(seq, max_l):
    """
    We pad to the longest sentence
    so returned vector all has the same length
    (but training set and testing set have different length)
    :return:
    """
    x = []
    x.extend(seq)
    while len(x) < max_l:
        x.append(0)
    return x


def padding(seqs, max_l):
    """
    :param seqs: list of vectors
    :param max_l: maximum length of a sentence/paragraph
    :param k: word vector
    :param filter_h: the highest width of a filter
    :return:
    """
    result = []
    for seq in seqs:
        result.append(padding_single(seq, max_l))
    return result


def sentence_filtering(train_set, maxlen):
    """
    run this function before padding
    we filter out sentences that are longer than maxLen
    modeled similar to dataloader.py in LSTM
    :return:
    """
    new_train_set_x = []
    new_train_set_y = []
    for x, y in zip(train_set[0], train_set[1]):
        if len(x) < maxlen:
            new_train_set_x.append(x)
            new_train_set_y.append(y)
    train_set = (new_train_set_x, new_train_set_y)
    del new_train_set_x, new_train_set_y

    return train_set


def sentence_clipping(sentence_set, maxLen=200):
    """
    This clips the sentence to a max length
    originally length 2000 something is too slow to train on
    clip starts from the tail
    :param sentence_set: this should be something like train_set_x
    :param maxLen: 200
    :return:
    """
    new_set = []
    for sentence in sentence_set:
        new_set.append(sentence[:maxLen])
    return new_set


def load_data(path, valid_portion=0.1, maxlen=200, permutation=False):
    """
    adapted from LSTM, read in like LSTM
    the only difference: we are doing wide-convolution, so need
    to add padding.

    permutation: whether want to randomly shuffle training/valid/testing data each time
                turn this off if we want to compare adv/no_adv

    :return: a list of vectors that is padded and has index as their [i]
    """

    #############
    # LOAD DATA #
    #############

    f = open(path)
    train_set = pkl.load(f)
    test_set = pkl.load(f)
    f.close()

    # we don't have maxlen here
    # split training set into validation set
    train_set_x, train_set_y = sentence_filtering(train_set, maxlen=maxlen)

    # get max length of training set
    lengths = [len(s) for s in train_set_x]
    train_maxlen = np.max(lengths)  # should be 2634 # but after filtering, it should be 100

    n_samples = len(train_set_x)  # after maxlen=100, we get 2444 samples
    n_train = int(np.round(n_samples * (1. - valid_portion)))

    if permutation:
        sidx = np.random.permutation(n_samples)
        valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
        valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
        train_set_x = [train_set_x[s] for s in sidx[:n_train]]
        train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    else:
        valid_set_x = train_set_x[n_train:]
        valid_set_y = train_set_y[n_train:]
        train_set_x = train_set_x[:n_train]
        train_set_y = train_set_y[:n_train]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    test_set_x, test_set_y = sentence_filtering(test_set, maxlen=maxlen)
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    test_lengths = [len(s) for s in test_set_x]
    test_maxlen = np.max(test_lengths)

    # let's do padding (for filter) here
    train_set_x = padding(train_set_x, train_maxlen)
    valid_set_x = padding(valid_set_x, train_maxlen)
    test_set_x = padding(test_set_x, test_maxlen)

    return {
        'X_train': np.asarray(train_set_x, dtype=theano.config.floatX),
        'y_train': np.asarray(train_set_y, dtype='int32'),
        'X_val': np.asarray(valid_set_x, dtype=theano.config.floatX),
        'y_val': np.asarray(valid_set_y, dtype='int32'),
        'X_test': np.asarray(test_set_x, dtype=theano.config.floatX),
        'y_test': np.asarray(test_set_y, dtype='int32')
    }


def load_idx_map(path):
    f = open(path, 'rb')
    word_emb, word_idx_map = pkl.load(f)
    # add a <NULL> token for padding (padding is always indexed as 0)
    word_idx_map['<NULL>'] = 0
    f.close()
    idx_word_map = dict((v, k) for k, v in word_idx_map.iteritems())
    return word_emb, word_idx_map, idx_word_map
