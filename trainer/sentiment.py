"""
A sentiment analysis trainer
that works on IMDB dataset
(almost exact same code from .ipynb)
in a file form so it can be run remotely
"""

from classifier.rnn_layers import *
from classifier.util import *
from data.sentiment_util import *
import numpy as np
from classifier.rnn_encoder_solver import EncoderSolver


data = load_data('data/rt_sentiment_data.npz', 'data/sentiment_vocab.json')

batch_size = 50

encoder = RNNEncoder(data['word_idx_map'], data['idx_word_map'], data['W_embed'], max_seq_length=57, batch_size=batch_size)

X = T.imatrix('X')
y = T.ivector('y')

solver = EncoderSolver(encoder,
                       data, X, y,
                       num_epochs=500, batch_size=batch_size,
                       update_rule='adam',
                       optim_config={
                           'learning_rate': 1e-2,
                       },
                       verbose=True, print_every=500)

solver.train()