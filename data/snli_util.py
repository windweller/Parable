"""
It loads the npz file, not the corpus
"""

import numpy as np


def load_data(path_to_npz):
    """
    Solver will take this object
    and work it out inside
    """
    # ['train_sentences', 'W_embed', 'dev_sentences', 'test_sentences',
    # 'y_dev', 'y_train', 'word_idx_map', 'y_test', 'idx_word_map']
    data = np.load(path_to_npz)

    # we need to store those into Python variables (into memory)

    print "loading training sentences..."
    train_sentences = data['train_sentences']

    print "loading dev sentences..."
    dev_sentences = data['dev_sentences']

    print "loading test sentences..."
    test_sentences = data['test_sentences']

    W_embed = data['W_embed']

    y_dev = data['y_dev']
    y_test = data['y_test']
    y_train = data['y_train']

    word_idx_map = data['word_idx_map']
    idx_word_map = data['idx_word_map']

    return {
        'train_sentences': train_sentences,
        'W_embed': W_embed, 'dev_sentences': dev_sentences,
        'test_sentences': test_sentences,
        'y_dev': y_dev, 'y_train': y_train, 'word_idx_map': word_idx_map,
        'y_test': y_test, 'idx_word_map': idx_word_map
    }
