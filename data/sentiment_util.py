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
    X_train = data['X_train']

    print "loading dev sentences..."
    X_val = data['X_val']

    print "loading test sentences..."
    X_test = data['X_test']

    W_embed = data['W_embed']

    y_dev = data['y_dev']
    y_test = data['y_test']
    y_train = data['y_train']

    word_idx_map = data['word_idx_map']
    idx_word_map = data['idx_word_map']

    return {
        'X_train': X_train,
        'W_embed': W_embed, 'X_val': X_val,
        'X_test': X_test,
        'y_dev': y_dev, 'y_train': y_train, 'word_idx_map': word_idx_map,
        'y_test': y_test, 'idx_word_map': idx_word_map
    }
