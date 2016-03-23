from layers import *


class ConvNet(object):
    """
    Very similar to CS231N's creation
    """

    def __init__(self, input_dim=(1, 32, 32),
                 batch_size=-100,
                 weight_scale=1e-3, reg=0.001,
                 dtype='float32'):
        """
        Initialize a new network. We create and store params here,
        and distribute on layers

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
            C: 1
            H: word_vector dim
            W: total word in a sentence (should be always the same)
            (This is VERY CRUCIAL, not to get it wrong)

        - batch_size: N value, which is passed in by Solver
                    Initialized as a negative value

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
        self.updates = []  # [(key, value)] this carries out non-parameter update rules (for batch norm)
        self.reg = reg
        self.dtype = dtype
        self.batch_size = batch_size  # this is N value
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

    def loss(self, X, y=None, final_loss=True):
        """
        This processes each stored layer and build up
        the computational graph

        We must assume a symbolic x and y, and they are not passing in
        Inputs:

        - X: symbolic expression T.matrix('x')
        - y: symbolic expression T.ivector('y')

        -final_loss: if True, we compute the final loss
                     if False, we can compute final loss outside

        Returns:
        - loss: symbolic expression, we no longer return grad
        """

        mode = 'test' if y is None else 'train'

        # we need to clean out self.updates, if loss() is called again
        if len(self.updates) != 0:
            self.updates = []

        # if test, we override bn_param to 'test'
        for param_dict in self.layer_param:
            if 'bn_param' in param_dict:
                param_dict['bn_param']['mode'] = mode

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
                t_return = spatial_batch_norm_layer(out, self.layer_param[i]['x_shape'],
                                                    self.params['gamma' + str(i)],
                                                    self.params['beta' + str(i)],
                                                    self.layer_param[i]['mean'],
                                                    self.layer_param[i]['var'],
                                                    self.layer_param[i]['bn_param'])
                if mode == 'train':
                    out = t_return[0]
                    self.updates.extend(t_return[1].items())
                elif mode == 'test':
                    out = t_return

            elif label is conv_batch_relu:
                out = conv_layer(out, self.params['W' + str(i)],
                                 self.params['b' + str(i)],
                                 self.layer_param[i]['conv_param'])
                t_return = spatial_batch_norm_layer(out, self.layer_param[i]['x_shape'],
                                                    self.params['gamma' + str(i)],
                                                    self.params['beta' + str(i)],
                                                    self.layer_param[i]['mean'],
                                                    self.layer_param[i]['var'],
                                                    self.layer_param[i]['bn_param'])
                if mode == 'train':
                    out = t_return[0]
                    self.updates.extend(t_return[1].items())
                elif mode == 'test':
                    out = t_return

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

        # if y is None:
        #     return out  # those are the final scores (we don't need to pass into softmax)

        # an if in case we use other final algorithm like SVM
        if self.layer_label[-1] is affine_softmax:
            out = softmax_layer(out)  # this gets us the softmax result
            # probabilities (or test predictions)

        # we need to pass into cross_entropy loss outside!!
        if final_loss and y is not None:
            out = -T.mean(T.log(out)[T.arange(y.shape[0]), y])  # compare test predictions with truth labels
            # out = -T.mean(T.nnet.categorical_crossentropy(out, y))

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

        # NOTICE: we are not using bias in Theano's conv2d implementation
        # UPDATE: we are using bias apostori

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
        """

        gamma = np.ones(number_filter, dtype=self.dtype)
        beta = np.zeros(number_filter, dtype=self.dtype)

        self.params['gamma' + str(self.i)] = self.wrap_shared_var(gamma,
                                                                  'gamma' + str(self.i),
                                                                  borrow=True)
        self.params['beta' + str(self.i)] = self.wrap_shared_var(beta,
                                                                 'beta' + str(self.i),
                                                                 borrow=True)

        mean = self.wrap_shared_var(beta, 'mean' + str(self.i),
                                    borrow=True)

        var = self.wrap_shared_var(beta, 'var' + str(self.i),
                                   borrow=True)

        bn_param = {'mode': 'train'}

        if self.batch_size <= 0:
            raise ValueError('Batch size cannot be negative: %s' % self.batch_size)

        # add x shape info
        # TODO: check if this is correct lol
        x_shape = (self.batch_size,
                   self.prev_depth, self.affine_H, self.affine_W)

        if len(self.layer_param) == self.i:
            # meaning: there is nothing in layer_param yet
            # need to check if this part is working :(
            self.layer_param.append({'bn_param': bn_param,
                                     'x_shape': x_shape,
                                     'mean': mean,
                                     'var': var})
        else:
            self.layer_param[self.i]['bn_param'] = bn_param
            self.layer_param[self.i]['x_shape'] = x_shape
            self.layer_param[self.i]['mean'] = mean
            self.layer_param[self.i]['var'] = var

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
