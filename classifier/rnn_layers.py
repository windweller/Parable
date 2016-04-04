"""
RNN layers ported from CS231N
"""

import numpy as np
import theano
import theano.tensor as T


def rnn_step_forward(x, prev_h, Wx, Wh, b, t=0, all_h=None):
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

    - t: current time_step (used by all_h)
    - all_h: a cumulative hidden states, of shape (N, T, H)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """

    # formula: h_t = f_W(h_t_1, x_t):
    # h_t = tanh(W_hh * h_t_1 + W_xh * x_t)
    # y = W_hy * h_t

    # store this for derivative
    next_h = T.tanh(T.dot(prev_h, Wh) + T.dot(x, Wx) + b)

    # all_h is None only when we are testing
    if all_h is not None:
        all_h = T.inc_subtensor(all_h[:, t, :], next_h)
        return next_h, all_h

    return next_h


def rnn_forward(x, h0, Wx, Wh, b, h_states_shapes):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
         N: batch-size
         T: sequence length (time-series size)
         D: dimension of each element in sequence (word vector length)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    - h_states_shapes: (N, T, H), shape for the h_states

    Returns a tuple of:
    - h_states: Hidden states for the entire timeseries, of shape (N, T, H).
      right now due to scan() limitation, we get (T, N, H)

    - updates: I don't know if updates needed to be passed back or not.
               It says they are for shared variables...?
    """

    # but T is already theano tensor

    N, TS, H = h_states_shapes

    # instead of for-loop, we use theano scan
    # we need updates, because updates are for shared_varaibles
    # and those do need to be updated

    # sequences are being iterated over, and pass in one at a time
    # outputs_info is passed in right after sequence, and is compounded
    # it initializes the argument.

    [_, all_ht_s], _ = theano.scan(
        fn=lambda t, h_t, all_h: rnn_step_forward(x[:, t, :], h_t, Wx, Wh, b, t, all_h),
        sequences=T.arange(TS),
        outputs_info=[h0, T.zeros((N, TS, H))])

    all_h_states = all_ht_s[-1]  # we only need the last one
    # because all_ht_s is a list of h_states of shape (T, N, H), but only at the last step
    # is this (T, N, H) completely filled up. So we only return the last one

    # h_ts is not needed because the shape is (T, N, H), not the (N, T, H) shape we are after

    return all_h_states


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b, H, t=0, all_h=None):
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

    # all_h is None only when we are testing: i.e.,
    # calling this function as a stand-alone function
    if all_h is not None:
        all_h = T.inc_subtensor(all_h[:, t, :], next_h)
        return next_h, next_c, all_h

    return next_h, next_c


def lstm_forward(x, h0, Wx, Wh, b, h_states_shapes):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    - h_states_shapes: dimension of hidden state (N, TS, H)

    Returns a tuple of:
    - hs: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    """

    N, TS, H = h_states_shapes

    # cell state is internal to LSTM, and not visible from outside
    [_, _, all_ht_s], _ = theano.scan(
        fn=lambda t, h_t, c_t, all_h: lstm_step_forward(x[:, t, :], h_t, c_t, Wx, Wh, b, H, t, all_h),
        sequences=T.arange(TS),
        outputs_info=[h0, np.zeros_like(h0, dtype=h0.dtype), T.zeros((N, TS, H))])

    hs = all_ht_s[-1]
    # we return all the hidden states along the time sequence
    # as this is needed for decoder or attention layer
    return hs


def lstm_inner_attention_layer(c_s):
    """
    This is embedded in LSTM, and only
    available to LSTM

    Inputs:
    - c_s: cell states input of shape (N, T, H)
            we call it a "memory band", and at
            each LSTM step, they can choose to READ
            any of the previous memory.

    Returns:
    - hs: modified hidden states
    """
    pass


def global_attention_layer(source_hs, target_hs):
    """
    Inputs:
    - source_hs: hidden states from the source
    - target_hs: hidden states from the target

    Returns:
    - m_target_hs: modified target hidden states
    """
    pass


def temporal_affine_forward(x, w, b, x_shape, b_shape):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Note: we explicitly pass in shape information from outside

    Inputs:
    - x: Input data of shape (N, T, D), must be a theano tensor
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """

    N, T, D = x_shape
    M = b_shape
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b

    return out


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Note: when we need to update W, we need to directly add W to param list.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """

    out = W[x]

    return out
