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

conv_relu = "conv_relu"
conv = "conv"
max_pool = "max_pool"
affine_relu = "affine_relu"
affine_softmax = "affine_softmax"


def leaky_ReLU(x, a=5.5):
    out = T.maximum(x, 0.0)
    mask = x < 0.0
    leaking_x = x * mask  # elem-wise, leave negative elems
    out += leaking_x * (1 / a)
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


def softmax_layer(x, y):
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
    loss = -T.mean(T.log(probs)[T.arange(y.shape[0]), y])
    return loss


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

    return out


def pooling_layer(x, pool_param):
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


class ConvNet(object):
    """
    Very similar to CS231N's creation
    """

    def __init__(self, input_dim=(1, 32, 32),
                 weight_scale=1e-3, reg=0.001,
                 dtype=theano.config.floatX):
        """
        Initialize a new network. We create and store params here,
        and distribute on layers

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
            C: 1
            H: word_vector dim
            W: total word in a sentence (should be always the same)
        - hidden_dim: Number of units to use in the fully-connected hidden layer
            We only have one in the end
        - num_classes: Number of scores to produce from the final affine layer.
            In sentiment analysis, and because of how we processed the data,
            we only have 2 classes
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
            We did not use Glorot because this network is incredibly shallow.
            For deeper network, we will use batchnorm
        - reg: Scalar giving L2 regularization strength
        - dtype: theano datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.input_dim = input_dim
        self.affine_H = input_dim[1]  # always record the "latest" affine shape
        self.affine_W = input_dim[2]  # same here, always updated to the latest
        self.prev_depth = input_dim[0]  # the C (channel) from x, but then will just be num_filter
        self.layer_label = []  # "cnn_batch_relu", "cnn_batch", "max_pool", "affine_batch_relu", "affine_softmax"
        self.layer_param = []  # index match layer_label
        self.weight_scale = weight_scale
        self.reg = reg
        self.i = 0  # number of layers, every time add one after using it

    def initialize(self):
        """
        This must (or just should) be called before running
        Returns:
        """
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(self.dtype)

    def loss(self, x, y):
        """
        This processes each stored layer and build up
        the computational graph

        We must assume a symbolic x and y, and they are not passing in
        Inputs:
        - x: symbolic expression T.matrix('x')
        - y: symbolic expression T.ivector('y')

        Returns:
        - loss: symbolic expression, we no longer return grad
        """

    def add_affine_relu_layer(self, prev_dim, hidden_dim):
        """
        Args:
            prev_dim: the dimension of previous layer
            hidden_dim: dim of hidden layer

        Returns:

        """
        self.params['W' + str(self.i)] = theano.shared(
            value=self.weight_scale * np.asarray(np.random.randn(
                prev_dim, hidden_dim), dtype=self.dtype),
            name='W' + str(self.i),
            borrow=True
        )
        self.params['b' + str(self.i)] = theano.shared(
            value=np.zeros(hidden_dim, dtype=self.dtype),
            name='b' + str(self.i),
            borrow=True
        )
        self.layer_label.append(affine_relu)
        self.layer_param.append({})
        self.i += 1

    def add_affine_softmax(self, prev_dim, num_classes):
        self.params['W' + str(self.i)] = theano.shared(
            value=self.weight_scale * np.asarray(np.random.randn(
                prev_dim, num_classes), dtype=self.dtype),
            name='W' + str(self.i),
            borrow=True
        )
        self.params['b' + str(self.i)] = theano.shared(
            value=np.zeros(num_classes, dtype=self.dtype),
            name='b' + str(self.i),
            borrow=True
        )

        self.layer_label.append(affine_softmax)
        self.layer_param.append({})
        self.i += 1

    def add_conv_layer(self, number_filter, filter_size, pad, stride):
        self.params['W' + str(self.i)] = theano.shared(
            value=self.weight_scale * np.asarray(np.random.randn(
                number_filter,
                self.prev_depth,
                filter_size, filter_size), dtype=self.dtype),
            name='W' + str(self.i),
            borrow=True
        )
        self.params['b' + str(self.i)] = theano.shared(
            value=np.zeros(number_filter, dtype=self.dtype),
            name='b' + str(self.i),
            borrow=True
        )
        self.prev_depth = number_filter
        self.affine_H = 1 + (self.affine_H + 2 * pad - filter_size) / stride
        self.affine_W = 1 + (self.affine_W + 2 * pad - filter_size) / stride

        conv_param = {'stride': stride, 'pad': pad}

        self.layer_label.append(conv)
        self.layer_param.append(conv_param)
        self.i += 1

    def add_pool_layer(self, size):
        self.affine_H /= size
        self.affine_W /= size
        self.layer_label.append(max_pool)
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': size, 'pool_width': size}
        self.layer_param.append({"pool_param": pool_param})

        self.i += 1
