"""
A sentiment analysis trainer
that works on IMDB dataset
(almost exact same code from .ipynb)
in a file form so it can be run remotely
"""
from parable.classifier.rnn_layers import *
from parable.classifier.util import *
from parable.data.sentiment_util import *
import numpy as np
from parable.classifier.rnn_encoder_solver import EncoderSolver
import os

pwd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

data = load_data(pwd+'/data/rt_sentiment_data.npz', pwd+'/data/sentiment_vocab.json')

num_train = 9500
batch_size = 50
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'][:500],
  'y_val': data['y_val'][:500],
}

X = T.imatrix('X')
y = T.ivector('y')

# max_seq_length = 57
encoder = RNNEncoder(data['word_idx_map'], data['idx_word_map'], data['W_embed'], max_seq_length=57, batch_size=batch_size)
solver = EncoderSolver(encoder,
                       small_data, X, y,
                num_epochs=100, batch_size=batch_size,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-2,
                },
                verbose=True, print_every=100)
solver.train()