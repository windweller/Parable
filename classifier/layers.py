"""
This file contains all layers
that we have for our calculation
"""


import numpy as np
import sys
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from collections import OrderedDict

conv = "conv"
conv_relu = "conv_relu"
conv_batch = "conv_batch"
conv_batch_relu = "conv_batch_relu"
max_pool = "max_pool"
avg_pool = "avg_pool"
affine_relu = "affine_relu"
affine_softmax = "affine_softmax"


def leaky_relu(x, a=100):
    out = T.maximum(x, 0.0)
    mask = x < 0.0
    leaking_x = x * mask  # elem-wise, leave negative elems
    out += leaking_x * (1.0 / a)
    return out


def affine_layer(x, w, b):
    """
    Inputs:
    - x: A Theano tensor containing input data, of shape (N, d_1, ..., d_k)
    - w: A Theano tensor of weights, of shape (D, M)
    - b: A Theano tensor of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    flattened_x = T.flatten(x, outdim=2)  # reshape into N, D
    out = T.dot(flattened_x, w) + b

    return out


def batch_norm_layer(x, gamma, beta, mean, var, bn_param):
    """
     Forward pass for batch normalization.

     During training the sample mean and (uncorrected) sample variance are
     computed from minibatch statistics and used to normalize the incoming data.
     During training we also keep an exponentially decaying running mean of the mean
     and variance of each feature, and these averages are used to normalize data
     at test-time.

     At each timestep we update the running averages for mean and variance using
     an exponential decay based on the momentum parameter:

     running_mean = momentum * running_mean + (1 - momentum) * sample_mean
     running_var = momentum * running_var + (1 - momentum) * sample_var

     Note that the batch normalization paper suggests a different test-time
     behavior: they compute sample mean and variance for each feature using a
     large number of training images rather than using a running average. For
     this implementation we have chosen to use running averages instead since
     they do not require an additional estimation step; the torch7 implementation
     of batch normalization also uses running averages.

     Input:
     - x: Data of shape (N, D)
     - gamma: Scale parameter of shape (D,), must be a Theano shared variable
     - beta: Shift paremeter of shape (D,), must be a Theano shared variable
     - bn_param: Dictionary with the following keys:
       - mode: 'train' or 'test'; required
       - eps: Constant for numeric stability
       - momentum: Constant for running mean / variance.
       - running_mean: Array of shape (D,) giving running mean of features
       - running_var Array of shape (D,) giving running variance of features

     Returns a tuple of:
     - out: of shape (N, D) (Theano expression)
     """

    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    batch_mean = T.mean(x, axis=0, dtype='float32')
    batch_var = T.var(x, axis=0)

    # so symbolic computation can carry on
    running_mean = theano.clone(mean, share_inputs=False)
    running_var = theano.clone(var, share_inputs=False)

    out = None
    mean_var_update = OrderedDict()

    if mode == 'train':
        # Compute output
        xc = x - batch_mean
        std = np.sqrt(batch_var + eps)
        xn = xc / std
        out = gamma * xn + beta

        # Update running average of mean
        running_mean *= momentum
        running_mean += (1 - momentum) * batch_mean

        # Update running average of variance
        running_var *= momentum
        running_var += (1 - momentum) * batch_var
    elif mode == 'test':
        # Using running mean and variance to normalize
        std = np.sqrt(var + eps)
        xn = (x - mean) / std
        out = gamma * xn + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # store update in the update rule
    # it gets updated on the outside
    mean_var_update[mean] = T.cast(running_mean, dtype='float32')
    mean_var_update[var] = T.cast(running_var, dtype='float32')

    if mode == 'train':
        return out, mean_var_update
    if mode == 'test':
        return out


def spatial_batch_norm_layer(x, x_shape, gamma, beta, mean, var, bn_param):
    """
    Same implementation as in Torch
    Careful with Theano's dimshuffle

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - x_shape: a tuple of value (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    for normal x: (N, D)
    gamma: shape of (D)
    beta: shape of (D)

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass

    """
    out, mean_var_updates = None, None
    N, C, H, W = x_shape
    mode = bn_param['mode']

    # n_c = x.transpose(1, 0, 2, 3).reshape((C, N * H * W)).transpose()
    # out = out.T.reshape((C, N, H, W)).transpose(1, 0, 2, 3)

    original_x = theano.clone(x, share_inputs=False)
    n_c = x.dimshuffle(1, 0, 2, 3).flatten(ndim=2).T
    if mode == 'train':
        out, mean_var_updates = batch_norm_layer(n_c, gamma, beta, mean, var, bn_param)
    elif mode == 'test':
        out = batch_norm_layer(n_c, gamma, beta, mean, var, bn_param)
    out = out.T.reshape((C, N, H, W)).dimshuffle(1, 0, 2, 3)

    if mode == 'train':
        return out, mean_var_updates
    elif mode == 'test':
        return out


def softmax_layer(x):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = T.nnet.softmax(x)
    # loss = -T.mean(T.log(probs)[T.arange(y.shape[0]), y])
    return probs


def conv_layer(x, w, b, conv_param):
    """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of N data points, each with C channels, height H and width
        W. We convolve each input with F different filters, where each filter spans
        all C channels and has height HH and width HH.

        Input:
        - x: Input data of shape (N, C, H, W), Theano 4D tensor
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
          - 'pad': The number of pixels that will be used to zero-pad the input.

        Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
    """
    # N, C, H, W = x.shape
    # F, C, HH, WW = w.shape
    # pad = (filter_size - 1) / 2
    # H_prime = 1 + (H + 2 * pad - HH) / stride
    # W_prime = 1 + (W + 2 * pad - WW) / stride

    stride, pad = conv_param['stride'], conv_param['pad']
    out = conv2d(x, w, border_mode=(pad, pad), subsample=(stride, stride))
    out = out + b.dimshuffle('x', 0, 'x', 'x')
    return out


def max_pooling_layer(x, pool_param):
    """

    Args:
        x:
        pool_param:

    Returns:

    """
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    pool_out = downsample.max_pool_2d(x, (pool_width, pool_height), ignore_border=True)
    return pool_out


def avg_pooling_layer(x, pool_param):
    """
    Perform average pooling instead of max
    normally used at the end of a network
    Args:
        x:
        pool_param:

    Returns:

    """
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    pool_out = downsample.max_pool_2d(x, (pool_width, pool_height), ignore_border=True, mode='average_exc_pad')
    return pool_out
