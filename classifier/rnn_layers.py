"""
RNN layers ported from CS231N
"""

import numpy as np
import theano
import theano.tensor as T


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, {}

    # formula: h_t = f_W(h_t_1, x_t):
    # h_t = tanh(W_hh*h_t_1 + W_xh * x_t)
    # y = W_hy * h_t

    # store this for derivative
    next_h = T.tanh(T.dot(prev_h, Wh) + T.dot(x, Wx) + b)

    return next_h


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b, H):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    - H: H = prev_c.shape[1] (we can't get H anymore, must be passed in)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """

    # H = prev_c.shape[1]

    a = T.dot(x, Wx) + T.dot(prev_h, Wh) + b
    # np.dot(x, Wx) = (N, 4H) -- be extra careful that hidden state (H, H) might be inverted
    # output: (N, 4H)
    a_i = a[:, 0:H]
    a_f = a[:, H:H * 2]
    a_o = a[:, H * 2:H * 3]
    a_g = a[:, H * 3: H * 4]

    i = T.nnet.sigmoid(a_i)
    f = T.nnet.sigmoid(a_f)
    o = T.nnet.sigmoid(a_o)
    g = T.tanh(a_g)

    next_c = f * prev_c + i * g
    next_h = o * T.tanh(next_c)

    return next_h, next_c


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache