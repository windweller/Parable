"""
It loads the npz file, not the corpus
"""

import numpy as np


def load_data(path_to_npz):
    """
    Solver will take this object
    and work it out inside
    """
    # ['idx_word_map', 'data', 'W_embed', 'word_idx_map']
    return np.load(path_to_npz)
