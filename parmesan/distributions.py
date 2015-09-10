import math
import theano.tensor as T


c = - 0.5 * math.log(2*math.pi)
def lognormal(x, mean, sd):
    return c - T.log(T.abs_(sd)) - (x - mean)**2 / (2 * sd**2)

def lognormal2(x, mean, logvar):
    return c - logvar/2 - (x - mean)**2 / (2 * T.exp(logvar))

def logstandard_normal(x):
    return c - x**2 / 2