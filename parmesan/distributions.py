import math
import theano.tensor as T


c = - 0.5 * math.log(2*math.pi)
def log_normal(x, mean, sd):
    return c - T.log(T.abs_(sd)) - (x - mean)**2 / (2 * sd**2)

def log_normal2(x, mean, logvar):
    return c - logvar/2 - (x - mean)**2 / (2 * T.exp(logvar))

def log_stdnormal(x):
    return c - x**2 / 2

def log_bernoulli(x, p):
    return -T.nnet.binary_crossentropy(p, x)

def kl_normal2_stdnormal(mean, logvar):
    return -0.5*(1 + logvar - mean**2 - T.exp(logvar))
