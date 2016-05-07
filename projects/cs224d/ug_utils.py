import h5py
import random
import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
from collections import defaultdict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import os

'''
helper functions
'''

rng = np.random.RandomState(1234)
floatX = theano.config.floatX

# dataset

# XXX dataset parameters
MNIST_PATH = '../data/mnist.pkl.gz'
FREYFACES_PATH = '../data/freyfaces.pkl'

# XXX refactor / split this file up

def save_model_params(model, fname):
    f = h5py.File(fname, 'w')
    names = [str(p) for p in model.params]
    assert(len(set(names)) == len(names))
    for p in model.params:
        dset = f.create_dataset(str(p), data=p.get_value())
    f.close()

def load_model_params(model, fname):
    f = h5py.File(fname, 'r')
    names = [str(p) for p in model.params]
    for p, name in zip(model.params, names):
        p.set_value(f[name][:])
    f.close()

def ortho_init(ndim, ndim1, act=None, scale=None):
    assert(ndim == ndim1)
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(floatX)

def norm_init(nin, nout, act=None, scale=0.01):
    W = scale * np.random.randn(nin, nout)
    return W.astype(floatX)

def uniform_init(nin, nout, act=None, scale=0.08):
    W = np.random.uniform(low=-1 * scale, high=scale, size=nin*nout)
    W = W.reshape((nin, nout))
    return W.astype(floatX)

def torch_init(nin, nout, act=None, scale=None):
    scale = 1. / np.sqrt(nout)
    return uniform_init(nin, nout, scale=scale)

def glorot_init(n_in, n_out, act=None, scale=None):
    W_values = np.asarray(
        rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        #rng.randn(n_in, n_out) * 0.01,
        dtype=floatX
    )
    if act == T.nnet.sigmoid:
        W_values *= 4
    return W_values

def get_sequence_dropout_mask(shape, p, stocdrop=False):
    srng = RandomStreams(seed=np.random.randint(1e6))
    if not stocdrop:
        return srng.binomial(size=shape, p=1.0 - p, dtype=floatX) / (1.0 - p)
    else:
        # FIXME assumes shape of dim (time steps, batch size, hidden size)
        col_mask = srng.binomial(size=(shape[0], shape[1], 1), p=1.0 - p, dtype=floatX)
        mask = T.tile(col_mask, (1, 1, shape[2]))
        return mask

class Dropout:

    def __init__(self, inp, p):
        # NOTE need to set p to 0 during testing
        self.srng = RandomStreams(seed=np.random.randint(1e6))
        self.p = p
        self.inp = inp
        self.out = self.inp * self.srng.binomial(size=self.inp.shape, p=1.0 - self.p, dtype=floatX) / (1.0 - self.p)

def _linear_params(n_in, n_out, suffix, init=uniform_init, scale=None, bias=True, act=None):
    if scale == None:
        scale = float(os.environ.get('WEIGHT_SCALE', '0.1'))
    W = theano.shared(init(n_in, n_out, act=act, scale=scale), 'W_' + suffix, borrow=True)
    if bias:
        b = theano.shared(np.zeros((n_out,), dtype=floatX), 'b_' + suffix, borrow=True)
        return W, b
    else:
        return W

def to_one_hot(y, nclasses):
    if np.isscalar(y):
        ny = 1
        y = [y]
    else:
        ny = y.shape[0]
    y_1_hot = np.zeros((ny, nclasses), dtype=np.int32)
    for k in xrange(ny):
        y_1_hot[k, y[k]] = 1
    return y_1_hot

def load_dataset(dset='mnist'):
    if dset == 'mnist':
        import gzip
        f = gzip.open(MNIST_PATH, 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()
        data = {'train': train_set, 'valid': valid_set, 'test': test_set}
    elif dset == 'freyfaces':
        with open(FREYFACES_PATH, 'rb') as f:
            full_set = np.array(pickle.load(f), dtype=floatX)
        # shuffle dataset
        np.random.seed(1234)
        perm = np.random.permutation(full_set.shape[0])
        full_set = full_set[perm, :]
        # XXX couldn't find standard split, using random .8/.1/.1 split
        N_train = 1573
        N_valid = N_test = 196
        # no labels
        data = {'train': (full_set[:N_train, :], None),
                'valid': (full_set[N_train:N_train+N_valid, :], None),
                'test': (full_set[-N_test:, :], None)}
    else:
        raise RuntimeError('unrecognized dataset: %s' % dset)
    return data

def get_labeled_examples(x, y, num_per_class=10, seed=1234):
    np.random.seed(seed)
    splits = defaultdict(list)
    for k in xrange(x.shape[0]):
        splits[y[k]].append(k)
    x_l = list()
    y_l = list()
    for c in splits:
        inds = random.sample(splits[c], num_per_class)
        x_l.append(x[inds, :])
        y_l.append(y[inds])
    x_l = np.concatenate(x_l, axis=0)
    y_l = np.concatenate(y_l)
    return x_l, y_l

# costs

def kld_unit_mvn(mu, var):
    # KL divergence from N(0, I)
    return (mu.shape[1] + T.sum(T.log(var), axis=1) - T.sum(T.square(mu), axis=1) - T.sum(var, axis=1)) / 2.0

def log_diag_mvn(mu, var):
    def f(x):
        # expects batches
        k = mu.shape[1]
        logp = (-k / 2.0) * np.log(2 * np.pi) - 0.5 * T.sum(T.log(var), axis=1) - T.sum(0.5 * (1.0 / var) * (x - mu) * (x - mu), axis=1)
        return logp
    return f

# test things out

if __name__ == '__main__':
    f = log_diag_mvn(np.zeros(2), np.ones(2))
    x = T.vector('x')
    g = theano.function([x], f(x))
    print g(np.zeros(2))
    print g(np.random.randn(2))

    mu = T.vector('mu')
    var = T.vector('var')
    j = kld_unit_mvn(mu, var)
    g = theano.function([mu, var], j)
    print g(np.random.randn(2), np.abs(np.random.randn(2)))
