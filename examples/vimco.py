# Implements VIMCO
#   Mnih, Andriy, and Danilo J. Rezende. "Variational inference for Monte Carlo
#   objectives." arXiv preprint arXiv:1602.06725 (2016).
import theano
theano.config.floatX = 'float32'
import matplotlib
matplotlib.use('Agg')
import theano.tensor as T
import numpy as np
import lasagne
from parmesan.distributions import log_bernoulli
from parmesan.datasets import load_mnist_realval, load_mnist_binarized
import matplotlib.pyplot as plt
from lasagne.layers import *
from lasagne.nonlinearities import *
from parmesan.layers import BernoulliSampleLayer
import shutil
import os
import time
import operator
import argparse
from parmesan.utils import log_sum_exp, log_mean_exp


def plotsamples(outputfilename, samples, ext='png', imgrange=(0.0, 1.0)):
    range_min, range_max = imgrange
    n, h, w = samples.shape
    samples = (samples-range_min)/np.float((range_max-range_min))
    samples_pr_side = int(np.sqrt(n))
    canvas = np.zeros((h*samples_pr_side, samples_pr_side*w))
    idx = 0
    for i in range(samples_pr_side):
        for j in range(samples_pr_side):
            canvas[i*h:(i+1)*h, j*w:(j+1)*w] = np.clip(
                samples[idx], 1e-6, 1-1e-6)
            idx += 1
    plt.figure(figsize=(7, 7))
    plt.imshow(canvas, cmap='gray')
    plt.savefig(outputfilename+'.'+ext)
    plt.close()

filename_script = os.path.basename(os.path.realpath(__file__))
default_path = os.path.join("results", os.path.splitext(filename_script)[0])
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str,
        help="sampled or fixed binarized MNIST, sample|fixed", default="sample")
parser.add_argument("-eq_samples", type=int,
        help="number of samples for the expectation over q(z|x)", default=1)
parser.add_argument("-iw_samples", type=int,
        help="number of importance weighted samples", default=50)
parser.add_argument("-lr", type=float,
        help="learning rate", default=0.001)
parser.add_argument("-anneal_lr_factor", type=float,
        help="learning rate annealing factor", default=0.99)
parser.add_argument("-anneal_lr_epoch", type=float,
        help="larning rate annealing start epoch", default=500)
parser.add_argument("-outfolder", type=str,
        help="output folder", default=default_path)
parser.add_argument("-nhidden", type=int,
        help="number of hidden units in deterministic layers", default=200)
parser.add_argument("-nlatent", type=int,
        help="number of stochastic latent units", default=200)
parser.add_argument("-batch_size", type=int,
        help="batch size", default=24)
parser.add_argument("-nepochs", type=int,
        help="number of epochs to train", default=10000)
args = parser.parse_args()


iw_samples = args.iw_samples    # number of importance weighted samples
eq_samples = args.eq_samples    # number of samples for the expectation
                                # over E_q(z|x)
lr = args.lr
anneal_lr_factor = args.anneal_lr_factor
anneal_lr_epoch = args.anneal_lr_epoch
res_out = args.outfolder
nhidden = args.nhidden
latent_size = args.nlatent
dataset = args.dataset
batch_size = args.batch_size
num_epochs = args.nepochs

assert dataset in ['sample', 'fixed'], "dataset must be sample|fixed"

np.random.seed(1234)  # reproducibility

# SET UP LOGFILE AND OUTPUT FOLDER
if not os.path.exists(res_out):
    os.makedirs(res_out)

# write commandline parameters to header of logfile
args_dict = vars(args)
sorted_args = sorted(args_dict.items(), key=operator.itemgetter(0))
description = []
description.append('######################################################')
description.append('# --Commandline Params--')
for name, val in sorted_args:
    description.append("# " + name + ":\t" + str(val))
description.append('######################################################')

shutil.copy(os.path.realpath(__file__), os.path.join(res_out, filename_script))
logfile = os.path.join(res_out, 'logfile.log')
model_out = os.path.join(res_out, 'model')
with open(logfile, 'w') as f:
    for l in description:
        f.write(l + '\n')


sym_iw_samples = T.iscalar('iw_samples')
sym_eq_samples = T.iscalar('eq_samples')
sym_lr = T.scalar('lr')
sym_x = T.matrix('x')


def bernoullisample(x):
    return np.random.binomial(1, x, size=x.shape).astype(theano.config.floatX)

# LOAD DATA AND SET UP SHARED VARIABLES
if dataset is 'sample':
    print "Using real valued MNIST dataset to binomial sample dataset " \
          "after every epoch "
    train_x, train_t, valid_x, valid_t, test_x, test_t = load_mnist_realval()
    del train_t, valid_t, test_t
    preprocesses_dataset = bernoullisample
else:
    print "Using fixed binarized MNIST data"
    train_x, valid_x, test_x = load_mnist_binarized()
    preprocesses_dataset = lambda dataset: dataset  # just a dummy function

train_x = np.concatenate([train_x, valid_x])

train_x = train_x.astype(theano.config.floatX)
test_x = test_x.astype(theano.config.floatX)
num_features = train_x.shape[-1]

sh_x_train = theano.shared(preprocesses_dataset(train_x), borrow=True)
sh_x_test = theano.shared(preprocesses_dataset(test_x), borrow=True)

# MODEL SETUP
# Recognition model q(z|x)
l_in = InputLayer((None, num_features))
l_enc_h1 = batch_norm(
    DenseLayer(l_in, num_units=nhidden, name='Q_DENSE1'), name='Q_')
l_enc_h1 = batch_norm(
    DenseLayer(l_enc_h1, num_units=nhidden, name='Q_DENSE2'), name='Q_')
l_enc_h1 = batch_norm(
    DenseLayer(l_enc_h1, num_units=nhidden, name='Q_DENSE3'), name='Q_')
l_mu = DenseLayer(
    l_enc_h1, num_units=latent_size, nonlinearity=T.nnet.sigmoid, name='Q_MU')

l_z = BernoulliSampleLayer(
    mean=l_mu, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)

l_z_in = InputLayer((None, latent_size))

# Generative model q(x|z)
l_dec_h1 = batch_norm(
    DenseLayer(l_z_in, num_units=nhidden, name='P_DENSE3'), name='P_')
l_dec_h1 = batch_norm(
    DenseLayer(l_dec_h1, num_units=nhidden, name='P_DENSE2'), name='P_')
l_dec_h1 = batch_norm(
    DenseLayer(l_dec_h1, num_units=nhidden, name='P_DENSE1'), name='P_')
l_dec_x_mu = DenseLayer(
    l_dec_h1, num_units=num_features,
    nonlinearity=T.nnet.sigmoid, name='P_X_MU')

z_train, z_mu_train = get_output(
    [l_z, l_mu], sym_x, deterministic=False)
x_mu_train = get_output(l_dec_x_mu, z_train, deterministic=False)

z_eval, z_mu_eval = get_output(
    [l_z, l_mu], sym_x, deterministic=True)
x_mu_eval = get_output(l_dec_x_mu, z_eval, deterministic=True)

x_mu_sample = get_output(l_dec_x_mu, sym_x, deterministic=True)

params = get_all_params([l_dec_x_mu, l_z], trainable=True)
q_params = [p for p in params if 'Q_' in p.name]
p_params = [p for p in params if 'P_' in p.name]


def get_vimco_baseline(x, combine=log_sum_exp):
    """
    Implements Right term in equation 10 from VIMCO paper
    x should be a 3d matrix in logspace
    """
    K_f = T.cast(x.shape[2], 'float32')
    return T.log(1.0/(K_f-1.0)) + no_norm_log_baseline_matrix_3d_th(
        x, combine=log_sum_exp)


def no_norm_log_baseline_matrix_3d_th(x, combine=log_sum_exp):
    """
    Calculate baseline matrix on last dimension of
    3d matrix
    """
    bs, mc, iw = x.shape
    x_ = x.reshape((-1, iw))
    x_ = no_norm_log_baseline_matrix_2d_th(x_, combine=combine)
    return x_.reshape((bs, mc, iw))


def no_norm_log_baseline_matrix_2d_th(x, combine=log_sum_exp):
    """
    Calculate unnormalized baselines for VIMCO estimator, sum f, i noteq j
    """
    bs, iw = x.shape
    # create mask that corresponds to sum over i exept when i=j
    inv_mask = T.eye(iw).reshape((iw, 1, iw)).repeat(bs, axis=1)

    # working in log space to we need to subtract the mask
    x_masked = x.reshape((1, bs, iw)) - inv_mask*1000000.0
    m = combine(x_masked, axis=2)
    return m.transpose()


def lower_bound(z, z_mu, x_mu, x, eq_samples, iw_samples, epsilon=1e-6):
    from theano.gradient import disconnected_grad as dg
    # reshape the variables so batch_size, eq_samples and iw_samples are
    # separate dimensions
    z = z.reshape((-1, eq_samples, iw_samples, latent_size))
    x_mu = x_mu.reshape((-1, eq_samples, iw_samples, num_features))

    # prepare x, z for broadcasting
    # size: (batch_size, eq_samples, iw_samples, num_features)
    x = x.dimshuffle(0, 'x', 'x', 1)

    # size: (batch_size, eq_samples, iw_samples, num_latent)
    z_mu = z_mu.dimshuffle(0, 'x', 'x', 1)

    log_qz_given_x = log_bernoulli(z, z_mu, eps=epsilon).sum(axis=3)
    z_prior = T.ones_like(z)*np.float32(0.5)
    log_pz = log_bernoulli(z, z_prior).sum(axis=3)
    log_px_given_z = log_bernoulli(x, x_mu, eps=epsilon).sum(axis=3)

    # Calculate the LL using log-sum-exp to avoid underflow
    log_pxz = log_pz + log_px_given_z

    # L is (bs, mc) See definition of L in appendix eq. 14
    L = log_sum_exp(log_pxz - log_qz_given_x, axis=2) + \
        T.log(1.0/T.cast(iw_samples, 'float32'))

    grads_model = T.grad(-L.mean(), p_params)

    # L_corr should correspond to equation 10 in the paper
    L_corr = L.dimshuffle(0, 1, 'x') - get_vimco_baseline(
        log_pxz - log_qz_given_x)
    g_lb_inference = T.mean(T.sum(dg(L_corr) * log_qz_given_x) + L)
    grads_inference = T.grad(-g_lb_inference, q_params)

    grads = grads_model + grads_inference
    LL = log_mean_exp(log_pz + log_px_given_z - log_qz_given_x, axis=2)
    return (LL,
            T.mean(log_qz_given_x), T.mean(log_pz), T.mean(log_px_given_z),
            grads)

# LOWER BOUNDS
(LL_train,
 log_qz_given_x_train,
 log_pz_train,
 log_px_given_z_train,
 grads) = lower_bound(
    z_train, z_mu_train,
    x_mu_train, sym_x,
    eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)

(LL_eval,
 log_qz_given_x_eval,
 log_pz_eval,
 log_px_given_z_eval,
 _) = lower_bound(
    z_eval, z_mu_eval,
    x_mu_eval, sym_x,
    eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)

# some sanity checks that we can forward data through the model
# dummy data for testing the implementation
X = np.ones((batch_size, 784), dtype=theano.config.floatX)

print "OUTPUT SIZE OF l_z1 using BS=%d, latent_size=%d, sym_iw_samples=%d, sym_eq_samples=%d --"\
      % (batch_size, latent_size, iw_samples, eq_samples), \
    lasagne.layers.get_output(l_z, sym_x).eval(
        {sym_x: X, sym_iw_samples: np.int32(iw_samples),
         sym_eq_samples: np.int32(eq_samples)}).shape

params = p_params + q_params

clip_grad = 1
max_norm = 5
mgrads = lasagne.updates.total_norm_constraint(grads, max_norm=max_norm)
cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

updates = lasagne.updates.adam(
    cgrads, params, beta1=0.9, beta2=0.999, epsilon=1e-4, learning_rate=sym_lr)

# Helper symbolic variables to index into the shared train and test data
sym_index = T.iscalar('index')
sym_batch_size = T.iscalar('batch_size')
batch_slice = slice(sym_index * sym_batch_size,
                    (sym_index + 1) * sym_batch_size)

train_model = theano.function(
    [sym_index, sym_batch_size, sym_lr, sym_eq_samples, sym_iw_samples],
    LL_train,
    givens={sym_x: sh_x_train[batch_slice]},
    updates=updates)

test_model = theano.function(
    [sym_index, sym_batch_size, sym_eq_samples, sym_iw_samples], LL_eval,
    givens={sym_x: sh_x_test[batch_slice]})

x_sample = (np.random.random((100, latent_size)) > 0.5).astype('float32')
sample100_model = theano.function([sym_x], x_mu_sample)


# Training and Testing functions
def train_epoch(lr, eq_samples, iw_samples, batch_size):
    n_train_batches = train_x.shape[0] / batch_size
    costs = []
    for i in range(n_train_batches):
        cost_batch = train_model(i, batch_size, lr, eq_samples, iw_samples)
        costs += [cost_batch]
    return np.mean(costs)


def test_epoch(eq_samples, iw_samples, batch_size):
    n_test_batches = test_x.shape[0] / batch_size
    costs = []
    output = {'x': []}
    for i in range(n_test_batches):
        cost_batch = test_model(i, batch_size, eq_samples, iw_samples)
        costs += [cost_batch]
    return np.mean(costs), output

print "Training"

# TRAIN LOOP
# We have made some the code very verbose to make it easier to understand.
costs_train = []
costs_eval = []
for epoch in range(1, 1+num_epochs):
    start = time.time()
    np.random.shuffle(train_x)
    sh_x_train.set_value(preprocesses_dataset(train_x))
    cost_train_ = train_epoch(lr, eq_samples, iw_samples, batch_size)
    cost_eval_, output_eval = test_epoch(eq_samples, 50, batch_size)
    costs_train += [cost_train_]
    costs_eval += [cost_eval_]
    plotsamples(res_out + '/prior.png', sample100_model(
        x_sample).reshape((-1, 28, 28)))

    plt.plot(costs_train, label='train')
    plt.plot(costs_eval, label='eval')
    plt.legend(loc=2)
    plt.title('Lower bound')
    plt.savefig(res_out + '/lb.png')
    plt.close()

    l = "epoch %i\ttrain %f\eval%f" % (epoch, cost_train_, cost_eval_)
    print l
    with open(logfile, 'a') as f:
        f.write(l + "\n")

    if epoch >= anneal_lr_epoch:
        lr = lr * anneal_lr_factor
