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
from theano.tensor.nnet import conv2d, bn

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


def batch_norm_layer(x, gamma, beta, bn_param):
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
     - gamma: Scale parameter of shape (D,)
     - beta: Shift paremeter of shape (D,)
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

    batch_mean = T.mean(x, axis=0)
    batch_var = T.var(x, axis=0)

    running_mean = bn_param.get('running_mean', T.zeros_like(batch_mean))
    running_var = bn_param.get('running_var', T.zeros_like(batch_var))

    out = None

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
        std = np.sqrt(running_var + eps)
        xn = (x - running_mean) / std
        out = gamma * xn + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means (theano expressions) back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out


def spatial_batch_norm_layer(x, gamma, beta, bn_param):
    """
    Same implementation as in Torch
    Careful with Theano's dimshuffle

    Inputs:
    - x: Input data of shape (N, C, H, W)
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
    out = None
    # n_c = x.transpose(1, 0, 2, 3).reshape((C, N * H * W)).transpose()
    # out = out.T.reshape((C, N, H, W)).transpose(1, 0, 2, 3)

    n_c = x.dimshuffle(1, 0, 2, 3).flatten(outdim=2).T


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


def resnet_block_layer():
    """
    Forms a resnet block
    2 conv layers, with beginning connect to bottom

    Returns:
    """


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
            (This is VERY CRUCIAL, not to get it wrong)

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
        self.i = 0  # number of layers, every time add one after using it (IMPORTANT for solver!!)

    def initialize(self):
        """
        This must (or just should) be called before running
        Returns:
        """
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(self.dtype)

    def loss(self, X, y=None):
        """
        This processes each stored layer and build up
        the computational graph

        We must assume a symbolic x and y, and they are not passing in
        Inputs:
        - X: symbolic expression T.matrix('x')
        - y: symbolic expression T.ivector('y')

        Returns:
        - loss: symbolic expression, we no longer return grad
        """

        mode = 'test' if y is None else 'train'

        # if test, we override bn_param to 'test'

        out = X

        for i, label in enumerate(self.layer_label):
            if label is conv_relu:
                out = conv_layer(out, self.params['W' + str(i)],
                                 self.params['b' + str(i)],
                                 self.layer_param[i]['conv_param'])
                out = leaky_relu(out, self.layer_param[i]['relu_a'])

            elif label is conv_batch:
                out = conv_layer(out, self.params['W' + str(i)],
                                 self.params['b' + str(i)],
                                 self.layer_param[i]['conv_param'])
                out = batch_norm_layer(out, self.params['gamma' + str(i)],
                                       self.params['beta' + str(i)],
                                       self.layer_param[i]['bn_param'])

            elif label is conv_batch_relu:
                out = conv_layer(out, self.params['W' + str(i)],
                                 self.params['b' + str(i)],
                                 self.layer_param[i]['conv_param'])
                out = batch_norm_layer(out, self.params['gamma' + str(i)],
                                       self.params['beta' + str(i)],
                                       self.layer_param[i]['bn_param'])
                out = leaky_relu(out, self.layer_param[i]['relu_a'])

            elif label is avg_pool:
                out = avg_pooling_layer(out, self.layer_param[i]['pool_param'])

            elif label is max_pool:
                out = max_pooling_layer(out, self.layer_param[i]['pool_param'])

            elif label is affine_relu:
                out = affine_layer(out, self.params['W' + str(i)],
                                   self.params['b' + str(i)])
                out = leaky_relu(out, self.layer_param[i]['relu_a'])

            elif label is affine_softmax:
                out = affine_layer(out, self.params['W' + str(i)],
                                   self.params['b' + str(i)])
            else:
                print "unknown layer: " + label
                sys.exit(0)

        if y is None:
            return out  # those are the final scores

        # an if in case we use other final algorithm like SVM
        if self.layer_label[-1] is affine_softmax:
            out = softmax_layer(out, y)  # this gets us the final loss

        return out

    def add_affine_relu_layer(self, prev_dim, hidden_dim, relu_a=100):
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
        self.layer_param.append({'relu_a': relu_a})
        self.i += 1

    def add_affine_softmax(self, prev_dim, num_classes):
        """
        Parameters
        ----------
        prev_dim: num_filters * max_pooled_affine_H *
                                      max_pooled_affine_W
        """
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

    def add_conv_layer(self, number_filter, filter_size, pad, stride, inde_layer=True):
        """

        Args:
            inde_layer: When flagged True, self.i will add 1

        Returns:

        """
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

        if inde_layer:
            self.layer_label.append(conv)

        if len(self.layer_param) == self.i:
            # meaning: there is nothing in layer_param yet
            # need to check if this part is working :(
            self.layer_param.append({'conv_param': conv_param})
        else:
            self.layer_param[self.i]['conv_param'] = conv_param

        if inde_layer:
            self.i += 1

    def add_conv_relu_layer(self, number_filter, filter_size, pad, stride, relu_a, inde_layer=True):
        """

        Args:
            inde_layer: When flagged True, self.i will add 1

        Returns:

        """
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

        if inde_layer:
            self.layer_label.append(conv_relu)

        if len(self.layer_param) == self.i:
            # meaning: there is nothing in layer_param yet
            # need to check if this part is working :(
            self.layer_param.append({'conv_param': conv_param,
                                     'relu_a': relu_a})
        else:
            self.layer_param[self.i]['conv_param'] = conv_param
            self.layer_param[self.i]['relu_a'] = relu_a

        if inde_layer:
            self.i += 1

    def wrap_shared_var(self, numpy_var, name, borrow):
        return theano.shared(
            value=numpy_var,
            name=name,
            borrow=True
        )

    def add_leaky_relu_layer(self, relu_a):
        """
        ReLU layer's setting is just appending ReLU parameter: a
        """
        if len(self.layer_param) == self.i:
            # meaning: there is nothing in layer_param yet
            # need to check if this part is working :(
            self.layer_param.append({'relu_a': relu_a})
        else:
            self.layer_param[self.i]['relu_a'] = relu_a

    def add_batch_layer(self, number_filter):
        """
        BatchNorm is never an "independent" layer
        (meaning it doesn't have W or b)
        so we don't add label to anything

        Args:
            number_filter:

        """

        gamma = np.ones(number_filter, dtype=self.dtype)
        beta = np.zeros(number_filter, dtype=self.dtype)

        self.params['gamma' + str(self.i)] = self.wrap_shared_var(gamma,
                                                                  'gamma' + str(self.i),
                                                                  borrow=True)
        self.params['beta' + str(self.i)] = self.wrap_shared_var(beta,
                                                                 'beta' + str(self.i),
                                                                 borrow=True)
        bn_param = {'mode': 'train'}

        if len(self.layer_param) == self.i:
            # meaning: there is nothing in layer_param yet
            # need to check if this part is working :(
            self.layer_param.append({'bn_param': bn_param})
        else:
            self.layer_param[self.i]['bn_param'] = bn_param

    def add_conv_batch_relu_layer(self, number_filter, filter_size, pad, stride, relu_a):
        self.add_conv_layer(number_filter, filter_size, pad, stride, inde_layer=False)

        # add batch norm
        self.add_batch_layer(number_filter)

        # add leaky relu
        self.add_leaky_relu_layer(relu_a)

        self.layer_label.append(conv_batch_relu)
        self.i += 1

    def add_conv_batch_layer(self, number_filter, filter_size, pad, stride):
        self.add_conv_layer(number_filter, filter_size, pad, stride, inde_layer=False)
        self.add_batch_layer(number_filter)
        self.layer_label.append(conv_batch)
        self.i += 1

    def add_res_block(self):
        """
        Res_block is composed of 2 conv layers
        of same size
        """
        input_depth_dim = self.prev_depth

    def add_jump_connect_res_block(self):
        """
        This is a custom resnet block.
        """
        pass

    def add_pool_layer(self, size, mode='max'):
        """
        Args:
            size: shrink size, size = 2, means shrink by half
            mode: 'max' or 'avg'
        """
        self.affine_H /= size
        self.affine_W /= size
        if mode == 'max':
            self.layer_label.append(max_pool)
        elif mode == 'avg':
            self.layer_label.append(avg_pool)
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': size, 'pool_width': size}
        self.layer_param.append({"pool_param": pool_param})

        self.i += 1
