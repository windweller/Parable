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
    next_h = np.tanh(np.dot(prev_h, Wh) + np.dot(x, Wx) + b)

    cache['hs_t'] = next_h
    cache['prev_hs'] = prev_h
    cache['x'] = x
    cache['Wh'] = Wh  # Whh (weight of hidden state)
    cache['Wx'] = Wx

    return next_h, cache