"""
It loads the npz file, not the corpus
"""

import numpy as np
import json


def load_data(path_to_npz, vocab_file_path):
    """
    Solver will take this object
    and work it out inside
    """
    # ['train_sentences', 'W_embed', 'dev_sentences', 'test_sentences',
    # 'y_dev', 'y_train', 'word_idx_map', 'y_test', 'idx_word_map']
    data = np.load(path_to_npz)

    # we need to store those into Python variables (into memory)

    print "loading source sentences..."
    train_src_sentences = data['train_src_sentences']
    dev_src_sentences = data['dev_src_sentences']
    test_src_sentences = data['test_src_sentences']

    print "loading target sentences..."

    train_tgt_sentences = data['train_tgt_sentences']
    dev_tgt_sentences = data['dev_tgt_sentences']
    test_tgt_sentences = data['test_tgt_sentences']

    y_train = data['y_train']
    y_dev = data['y_dev']
    y_test = data['y_test']

    print "loading embeddings..."

    W_embed = data['W_embed']

    with open(vocab_file_path, 'r') as f:
        vocab = json.load(f)
        word_idx_map = vocab['word_idx_map']
        idx_word_map = vocab['idx_word_map']

    return {
        'train_src_sentences': train_src_sentences,
        'W_embed': W_embed, 'dev_src_sentences': dev_src_sentences,
        'test_src_sentences': test_src_sentences,
        'train_tgt_sentences': train_tgt_sentences,
        'dev_tgt_sentences': dev_tgt_sentences,
        'test_tgt_sentences': test_tgt_sentences,
        'y_dev': y_dev, 'y_train': y_train, 'word_idx_map': word_idx_map,
        'y_test': y_test, 'idx_word_map': idx_word_map
    }
