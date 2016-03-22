import optim
import numpy as np
import theano.tensor as T
import theano
from classifier.util import wrap_shared_var

"""
Solver will take in
data and has util functions
for checking accuracies
"""


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various optoins (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists containing
    the accuracies of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'X_train': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    A Solver works on a model object that must conform to the following API:

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.

    - model.updates must contain not-parameter update rules (such as Batch Normalization
      we will update running_mean and running_var)

    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:

      Inputs:
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
        label for X[i].

      Returns:
      If y is None, run a test-time forward pass and return:
      - scores: Array of shape (N, C) giving classification scores for X where
        scores[i, c] gives the score of class c for X[i].

      If y is not None, run a training time forward and backward pass and return
      a tuple of:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.
    """

    def __init__(self, model, data, symbolic_X, symbolic_y, **kwargs):
        """
            Construct a new Solver instance.

            Required arguments:
            - model: A model object conforming to the API described above
            - data: A dictionary of training and validation data with the following:
              'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
              'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
              'y_train': Array of shape (N_train,) giving labels for training images
              'y_val': Array of shape (N_val,) giving labels for validation images

            - symbolic_X: A theano symbolic variable like T.matrix('x') or T.tensor4('X')
                          that corresponds to the shape of X passing in
            - symbolic_y: A theano symbolic variable like T.ivector('y')
                          that corresponds to the shape of y passing in

            Optional arguments:
            - update_rule: A string giving the name of an update rule in optim.py.
              Default is 'sgd'.
            - optim_config: A dictionary containing hyperparameters that will be
              passed to the chosen update rule. Each update rule requires different
              hyperparameters (see optim.py) but all update rules require a
              'learning_rate' parameter so that should always be present.
            - lr_decay: A scalar for learning rate decay; after each epoch the learning
              rate is multiplied by this value.
            - batch_size: Size of minibatches used to compute loss and gradient during
              training.
            - num_epochs: The number of epochs to run for during training.
            - print_every: Integer; training losses will be printed every print_every
              iterations.
            - verbose: Boolean; if set to false then no output will be printed during
              training.
            """
        self.model = model
        self.X_train = np.asarray(data['X_train'], dtype='float32')
        self.y_train = np.asarray(data['y_train'], dtype='int32')
        self.X_val = np.asarray(data['X_val'], dtype='float32')
        self.y_val = np.asarray(data['y_val'], dtype='int32')
        self.X = symbolic_X  # data, presented as rasterized images
        self.y = symbolic_y  # labels, presented as 1D vector of [int] labels

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        # we don't know if _reset() is resetting everything
        self._reset()

        # constructing train_fn and valid_fn (also used for test_fn)
        self.train_loss = self.model.loss(self.X, self.y, final_loss=True)
        self.updates = []

        # Perform a parameter update (by param update)
        for p, w in self.model.params.iteritems():
            # p: name of param, w: actual param (shared variable) (x is not there)
            dw = T.grad(self.train_loss, w)
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.updates.append((self.model.params[p], next_w))  # we store those updates
            self.optim_configs[p] = next_config

        # ADD OTHER STUFF TO UPDATES list (such as BATCH NORM's updates)
        self.updates.extend(self.model.updates)

        self.train_fn = theano.function([self.X, self.y], self.train_loss, updates=self.updates)

        # However, our validation function follows a different style:
        # We take index slice as input, not actual batch as input

        self.test_scores = self.model.loss(self.X, final_loss=False)  # this is the softmax probability
        self.test_loss = T.nnet.categorical_crossentropy(self.test_scores, self.y)  # loss

        self.test_loss = self.test_loss.mean()
        self.test_acc = T.mean(T.eq(T.argmax(self.test_scores, axis=1), self.y),
                               dtype='float32')

        self.val_fn = theano.function([self.X, self.y], [self.test_scores, self.test_loss, self.test_acc])

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # clean neural network settings
        self.train_fn = None
        self.updates = []
        self.train_loss = None
        self.test_scores = None

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.iteritems()}
            self.optim_configs[p] = d

    def _step(self, t):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        This requires massive change.
        actual parameter updates are actually incurred inside train function

        we also must pass the loss function from the outside

        Returns
        - updates: [(original var, updated var)], to obey Theano's update rules
        """

        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # We create a Theano function
        # this gets the loss and run parameter update!

        # X_batch_shared = wrap_shared_var(X_batch, 'X_batch' + str(t), borrow=True)
        # y_batch_shared = wrap_shared_var(y_batch, 'y_batch' + str(t), borrow=True)

        num_loss = self.train_fn(X_batch, y_batch)

        self.loss_history.append(num_loss)

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using too
          much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # TODO: This needs rework...to fit Theano
        # if self.y is not passed in,
        # loss function will set bn_param to 'test'

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1

        # X_shared = wrap_shared_var(X, 'X_check_accuracy', borrow=True)
        # y_shared = wrap_shared_var(y, 'y_check_accuracy', borrow=True)

        # Compute test loss
        test_loss, test_acc = None, None
        for i in xrange(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            _, test_loss, test_acc = self.val_fn(X[start:end],
                                                 y[start:end])  # self.model.loss(X[start:end])

        return test_loss, test_acc

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in xrange(num_iterations):
            self._step(t)

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print '(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, self.loss_history[-1])

            # At the end of every epoch, increment the epoch counter and decay the
            # learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations + 1)

            if first_it or last_it or epoch_end:
                _, train_acc = self.check_accuracy(self.X_train, self.y_train, batch_size=self.batch_size) # , num_samples=1000
                val_loss, val_acc = self.check_accuracy(self.X_val, self.y_val, batch_size=self.batch_size)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print '(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                        self.epoch, self.num_epochs, train_acc, val_acc)

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.iteritems():
                        self.best_params[k] = v.get_value(borrow=False)  # TODO: this line might not work

        # At the end of training swap the best params into the model
        for k, v in self.model.params.iteritems():
            # borrow = True because we use numpy's buffer
            self.model.params[k].set_value(self.best_params[k], borrow=True)
