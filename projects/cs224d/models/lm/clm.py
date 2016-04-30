import time
import sys
import numpy as np
import cPickle as pickle
import theano
from theano import tensor as T
from parable.projects.cs224d.data.char_fuel import SimpleCharLoader, ShufflingCharLoader
from parable.projects.cs224d.models.run_utils import setup_exp
from os.path import join as pjoin
from parable.projects.cs224d.ug_utils import floatX, Dropout, get_sequence_dropout_mask
from parable.projects.cs224d.ug_cfg import (UG_DIR, CHAR_FILE, PTB_FILE)
from parable.projects.cs224d.models.rnn import (RNN, SequenceLogisticRegression, GRULayer, LSTMLayer,
                                                LayerWrapper, seq_cat_crossent)
from parable.projects.cs224d.opt import get_opt_fn, optimizers, clip_grads

'''
simple character language model sanity check

compare to: https://github.com/karpathy/char-rnn/blob/master/train.lua
'''

# FIXME pickling model currently requires this, but should instead just save/load parameters
sys.setrecursionlimit(100000)

UnitInit = {'gru': GRULayer, 'lstm': LSTMLayer}

def flatten(list_of_lists):
    return [x for sublist in list_of_lists for x in sublist]

class RNNLM(RNN):

    def __init__(self, args):
        self.args = args
        x = T.imatrix('x')
        y = T.imatrix('y')
        mask = T.ones_like(x).astype(floatX)
        # FIXME TODO resume from last state of previous sequence instead o
        # resetting the first hidden state to 0s
        self.unit = args.unit
        if args.unit == 'gru':
            init_states = [T.matrix(dtype=floatX) for k in xrange(args.rlayers)]
        elif args.unit == 'lstm':
            init_states = [(T.matrix(dtype=floatX), T.matrix(dtype=floatX)) for k in xrange(args.rlayers)]
        else:
            assert(False)
        lr = T.scalar(dtype=floatX)
        pdrop = T.scalar(dtype=floatX)

        rlayers = list()
        inp = theano.tensor.extra_ops.to_one_hot(x.flatten(), args.vocab_size).astype(floatX).reshape((x.shape[0], x.shape[1], args.vocab_size))
        seqmask = get_sequence_dropout_mask((inp.shape[0], inp.shape[1], args.rnn_dim), pdrop, stocdrop=args.stocdrop)
        # exclude last prediction
        inplayer = UnitInit[args.unit](inp.astype(floatX), mask, seqmask, args.vocab_size, init_states[0], args, suffix='0')
        rlayers.append(inplayer)
        for k in xrange(1, args.rlayers):
            seqmask = get_sequence_dropout_mask((inp.shape[0], inp.shape[1], args.rnn_dim), pdrop, stocdrop=args.stocdrop)
            rlayer = UnitInit[args.unit](Dropout(rlayers[-1].out, pdrop).out, mask, seqmask, args.rnn_dim, init_states[k], args, suffix='%d' % k)
            rlayers.append(rlayer)
        olayer = SequenceLogisticRegression(Dropout(rlayers[-1].out, pdrop).out, args.rnn_dim,
                args.vocab_size)
        self.cost = seq_cat_crossent(olayer.out, y, mask, normalize=False)
        super(RNNLM, self).__init__(rlayers, olayer, cost=self.cost)
        shapes = [p.shape.eval() for p in self.params]
        sizes = [np.prod(s) for s in shapes]
        self.nparams = np.sum(sizes)
        self.updates, self.grad_norm, self.param_norm = get_opt_fn(args.optimizer)(self.cost, self.params, lr, max_norm=args.max_norm)

        # functions

        if args.unit == 'lstm':
            init_states = flatten(init_states)
            final_states = list()
            for r in rlayers:
                final_states.append(r.out[-1])
                final_states.append(r.cell[-1])
        else:
            final_states = [r.out[-1] for r in rlayers]

        self.train = theano.function(
            inputs=[x, y, pdrop, lr] + init_states,
            outputs=[self.cost, self.grad_norm, self.param_norm] + final_states,
            updates = self.updates,
            on_unused_input='warn'
        )

        self.test = theano.function(
            # at test time should pass in pdrop=0
            inputs=[x, y, pdrop] + init_states,
            outputs=[self.cost] + final_states,
            updates = None,
            on_unused_input='warn'
        )

        # function for sampling

        i_t = T.ivector()
        x_t = theano.tensor.extra_ops.to_one_hot(i_t, args.vocab_size)[0]
        h_ps = list()  # previous
        for k in xrange(args.rlayers):
            if args.unit == 'gru':
                h_ps.append(T.vector())
                dmask = T.ones_like(h_ps[0]).astype(floatX)
            else:
                h_ps.append((T.vector(), T.vector()))
                dmask = T.ones_like(h_ps[0][0]).astype(floatX)
        h_ts = list()
        if args.unit == 'lstm':
            h_t = self.rlayers[0]._step(x_t, dmask, *h_ps[0])
        else:
            h_t = self.rlayers[0]._step(x_t, dmask, h_ps[0])
        h_ts.append(h_t)
        for k in xrange(1, args.rlayers):
            if args.unit == 'lstm':
                h_t = self.rlayers[k]._step(h_t[0], dmask, *h_ps[k])
            else:
                h_t = self.rlayers[k]._step(h_t, dmask, h_ps[k])
            h_ts.append(h_t)
        if args.unit == 'lstm':
            h_t = h_t[0]
        E_t = T.dot(h_t, self.olayer.W) + self.olayer.b
        E_t = T.exp(E_t - T.max(E_t))
        p_t = E_t / E_t.sum()
        if args.unit == 'lstm':
            h_ps = flatten(h_ps)
            h_ts = flatten(h_ts)
        self.decode_step = theano.function(
            inputs=[i_t] + h_ps,
            outputs=[p_t] + h_ts,
            updates=None,
            on_unused_input='warn'
        )

def sample_output(rnnlm, h_ps, fdict, bdict):
    sample = ''
    s_t = np.array([fdict['\n']], dtype=np.int32)  # have to start with something
    j = 0
    while j < rnnlm.args.unroll:
        ret = rnnlm.decode_step(*([s_t] + h_ps))
        p_t = ret[0]
        p_t = p_t.flatten() / (np.sum(p_t) + 1e-6)  # hack
        h_ps = ret[1:]
        s_t = np.array([np.argmax(np.random.multinomial(1, p_t))], dtype=np.int32)
        sample = sample + bdict[s_t[0]]
        j = j + 1
    return sample

if __name__ == '__main__':
    import json
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_dim', type=int, default=256, help='dimension of recurrent states')
    parser.add_argument('--rlayers', type=int, default=2, help='number of hidden layers for RNNs')
    parser.add_argument('--unroll', type=int, default=50, help='number of time steps to unroll for')
    parser.add_argument('--batch_size', type=int, default=50, help='size of batches')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--lr_decay_after', type=int, default=10, help='epoch after which to decay')
    #parser.add_argument('--max_norm_elemwise', type=float, default=0.1, help='norm at which to clip gradients elementwise')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout (fraction of units randomly dropped on non-recurrent connections)')
    parser.add_argument('--recdrop', action='store_true', help='use dropout on recurrent updates as well')
    parser.add_argument('--stocdrop', action='store_true', help='use in combination with --recdrop to actually drop entire update for certain (layer, time-step)s')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--max_norm', type=float, default=1.0, help='gradient clipping in 2-norm')
    parser.add_argument('--unit', type=str, choices=('gru', 'lstm'), default='gru')
    parser.add_argument('--print_every', type=int, default=1, help='how often to print cost')
    parser.add_argument('--optimizer', type=str, default='adam', choices=optimizers)
    parser.add_argument('--expdir', type=str, default='sandbox', help='experiment directory to save files to')
    parser.add_argument('--train_frac', type=float, default=0.95, help='fraction of text file to use for training data')
    parser.add_argument('--valid_frac', type=float, default=0.05, help='fraction of text file to use for validation data')
    parser.add_argument('--batch_norm', dest='batch_norm', action='store_true')
    parser.add_argument('--ortho', dest='ortho', action='store_true', help='Orthogonal Initialization')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for the pickle where costs are stored')
    parser.add_argument('--data', type=str, default='CHAR', help='CHAR=CHAR_FILE, PTB=PTB_FILE, specify to indicate corpus')

    parser.set_defaults(batch_norm=False)

    args = parser.parse_args()
    args.max_seq_len = args.unroll
    args.resume_epoch = None

    print args

    if args.batch_norm:
        print 'Doing batch norm'

    logger, opts = setup_exp(args)

    CORPUS = ""
    if args.data == 'CHAR':
        CORPUS = CHAR_FILE
    elif args.data == 'PTB':
        CORPUS = PTB_FILE

    loader = SimpleCharLoader(CORPUS, args.batch_size, args.max_seq_len,
        train_frac=args.train_frac, valid_frac=args.valid_frac)
    train_split = loader.splits['train']
    val_split = loader.splits['val']
    bdict = {v: k for k, v in loader.vocab.iteritems()}
    args.vocab_size = len(bdict)
    logger.info('vocab size: %d' % args.vocab_size)

    start_epoch = 0
    rnnlm = RNNLM(args)
    logger.info('# params: %d' % rnnlm.nparams)

    expcost = None
    lr = args.lr

    train_costs = []
    valid_costs = []

    for epoch in xrange(args.epochs):
        if epoch >= args.lr_decay_after and args.optimizer not in ['adam']:
            lr = lr * args.lr_decay

        it = 0
        init_states = [np.zeros((args.batch_size, args.rnn_dim), dtype=floatX)] * args.rlayers
        if args.unit == 'lstm':
            init_states = init_states * 2
        for k in xrange(loader.num_train_batches):
            tic = time.time()
            it = it + 1
            x, y = loader.get_batch('train', k)
            ret = rnnlm.train(x.T, y.T, args.dropout, lr, *init_states)
            cost, grad_norm, param_norm = ret[0:3]
            norm_ratio = grad_norm / param_norm
            cost = cost / args.max_seq_len
            train_costs.append(cost)
            init_states = ret[3:]
            if not expcost:
                expcost = cost
            else:
                expcost = 0.01 * cost + 0.99 * expcost
            toc = time.time()
            if (it + 1) % args.print_every == 0:
                logger.info('epoch %d, iter %d, cost %f, expcost %f, batch time %f, grad/param norm %f' %\
                        (epoch + 1, it, cost, expcost, toc - tic, norm_ratio))

        #loader.shuffle('train')
        # run on validation
        for k in xrange(loader.num_val_batches):
            x, y = loader.get_batch('val', k)
            # NOTE set dropout rate to 0
            ret  = rnnlm.test(x.T, y.T, 0.0, *init_states)
            cost = ret[0] / args.max_seq_len
            init_states = ret[1:]
            valid_costs.append(cost)

        #loader.shuffle('val')

        # sample some output
        h_ps = [np.zeros(args.rnn_dim).astype(floatX)] * args.rlayers
        if args.unit == 'lstm':
            h_ps = h_ps * 2
        for k in xrange(10):
            sample = sample_output(rnnlm, h_ps, loader.vocab, bdict)
            print(sample)

        logger.info('validation cost: %f' %\
                (sum(valid_costs) / float(len(valid_costs))))

        # save model # FIXME
        logger.info('saving model')

    with open('costs_%s.pkl' % args.suffix, 'wb') as f:
        pickle.dump((valid_costs, train_costs), f)

    with open(pjoin(args.expdir, 'model_epoch%d.pk' % epoch), 'wb') as f:
        pickle.dump(rnnlm, f, protocol=pickle.HIGHEST_PROTOCOL)
