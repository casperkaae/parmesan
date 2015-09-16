import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import lasagne
import numpy as np
from parmesan.distributions import logstandard_normal, lognormal2
from parmesan.layers import SampleLayer
from parmesan.datasets import load_mnist_realval
import time, shutil, os


#settings
batch_size = 250
latent_size = 250
lr = 0.005
num_epochs = 1000
results_out = 'results/vae_vanilla/'

# Setup outputfolder logfie etc.
if not os.path.exists(results_out):
    os.makedirs(results_out)
scriptpath = os.path.realpath(__file__)
filename = os.path.basename(scriptpath)
shutil.copy(scriptpath,results_out + filename)
logfile = results_out + 'logfile.log'
model_out = results_out + 'model'

#SYMBOLIC VARS
sym_x = T.matrix()
sym_lr = T.scalar('lr')


#Helper functions
def bernoullisample(x):
    return np.random.binomial(1,x,size=x.shape).astype(theano.config.floatX)


### LOAD DATA
train_x, train_t, valid_x, valid_t, test_x, test_t = load_mnist_realval()
#concatenate train and validation set
train_x = np.concatenate([train_x, valid_x])

nfeatures=train_x.shape[1]
n_train_batches = train_x.shape[0] / batch_size
n_test_batches = test_x.shape[0] / batch_size

#setup shared variables
sh_x_train = theano.shared(np.asarray(bernoullisample(train_x), dtype=theano.config.floatX), borrow=True)
sh_x_test = theano.shared(np.asarray(bernoullisample(test_x), dtype=theano.config.floatX), borrow=True)


### RECOGNITION MODEL q(z|x)
l_in = lasagne.layers.InputLayer((batch_size, nfeatures))
l_enc_h1 = lasagne.layers.DenseLayer(l_in, num_units=100, name='ENC_DENSE1')
l_enc_h1 = lasagne.layers.DenseLayer(l_enc_h1, num_units=100, name='ENC_DENSE2')

l_mu = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_MU')
l_log_var = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_LOG_VAR')

#sample the latent variables using mu(x) and log(sigma^2(x))
l_z = SampleLayer(mu=l_mu, log_var=l_log_var)

### GENERATIVE MODEL p(x|z)
l_dec_h1 = lasagne.layers.DenseLayer(l_z, num_units=100, name='DEC_DENSE2')
l_dec_h1 = lasagne.layers.DenseLayer(l_dec_h1, num_units=100, name='DEC_DENSE1')
l_dec_x_mu = lasagne.layers.DenseLayer(l_dec_h1, num_units=nfeatures, nonlinearity=lasagne.nonlinearities.sigmoid, name='DEC_Xmu')


# Get outputs from model
# with noise
z_train, z_mu_train, z_log_var_train, x_mu_train = lasagne.layers.get_output(
    [l_z, l_mu, l_log_var, l_dec_x_mu], sym_x, deterministic=False
)

# without noise
z_eval, z_mu_eval, z_log_var_eval, x_mu_eval = lasagne.layers.get_output(
    [l_z, l_mu, l_log_var, l_dec_x_mu], sym_x, deterministic=True
)


#Calculate the loglikelihood(x) = E_q[ log(x|z) + p(z) - q(z|x)]
def latent_gaussian_x_bernoulli(z, z_mu, z_log_var, x, x_mu):
    """
    Latent z       : gaussian with standard normal prior
    decoder output : bernoulli

    When the output is bernoulli then the output from the decoder
    should be sigmoid.
    """
    log_qz_given_x = lognormal2(z, z_mu, z_log_var).sum(axis=1)
    log_pz = logstandard_normal(z).sum(axis=1)
    log_px_given_z = T.sum(-T.nnet.binary_crossentropy(x_mu, x), axis=1)
    LL = T.mean(log_pz + log_px_given_z - log_qz_given_x)
    return LL

# TRAINING LogLikelihood
LL_train = latent_gaussian_x_bernoulli(
    z_train, z_mu_train, z_log_var_train, sym_x, x_mu_train)

# EVAL LogLikelihood
LL_eval = latent_gaussian_x_bernoulli(
    z_eval, z_mu_eval, z_log_var_eval, sym_x, x_mu_eval)


params = lasagne.layers.get_all_params([l_dec_x_mu], trainable=True)
for p in params:
    print p, p.get_value().shape

### Take gradient of Negative LogLikelihood
grads = T.grad(-LL_train, params)

# Add gradclipping to reduce the effects of exploding gradients.
# This speeds up convergence
clip_grad = 1
max_norm = 5
mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
cgrads = [T.clip(g,-clip_grad, clip_grad) for g in mgrads]


#Setup the theano functions
sym_batch_index = T.iscalar('index')
batch_slice = slice(sym_batch_index * batch_size, (sym_batch_index + 1) * batch_size)

updates = lasagne.updates.adam(cgrads, params, learning_rate=sym_lr)

train_model = theano.function([sym_batch_index, sym_lr], LL_train, updates=updates,
                                  givens={sym_x: sh_x_train[batch_slice], },)

test_model = theano.function([sym_batch_index], LL_train,
                                  givens={sym_x: sh_x_test[batch_slice], },)


def train_epoch(lr):
    costs = []
    for i in range(n_train_batches):
        cost_batch = train_model(i, lr)
        costs += [cost_batch]
    return np.mean(costs)


def test_epoch():
    costs = []
    for i in range(n_test_batches):
        cost_batch = test_model(i)
        costs += [cost_batch]
    return np.mean(costs)


# Training Loop
for epoch in range(num_epochs):
    start = time.time()
    train_cost = train_epoch(lr)
    test_cost = test_epoch()
    t = time.time() - start
    line =  "*Epoch: %i\tTime: %0.2f\tLR: %0.5f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, lr, train_cost, test_cost)
    print line
    with open(logfile,'a') as f:
        f.write(line + "\n")







