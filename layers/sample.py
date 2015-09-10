import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
import theano
import numpy as np

class SampleLayer(lasagne.layers.MergeLayer):
    """
    Simple sampling layer drawing a single monte carlo sample to approximate
    E_q [log( p(x,z) / q(z|x) )]. This is the approach desribed in
    Kingma et. al. 2013 [KINGMA]

    Parameters
    ----------
    mu, log_var : class:`Layer` instances
        Parametrizing the mean and log(variance) of the distribution to sample
        from as described in [KINGMA]. The code assumes that these have the same
        number of dimensions


    References
        [KINGMA] Kingma, Diederik P., and Max Welling.
        "Auto-encoding variational bayes."
        arXiv preprint arXiv:1312.6114 (2013).

    """
    def __init__(self, mu, log_var, **kwargs):
        super(SampleLayer, self).__init__([mu, log_var], **kwargs)

        self._srng = RandomStreams(
            lasagne.random.get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        mu, log_var = input
        eps = self._srng.normal(mu.shape)
        z = mu + T.exp(0.5 * log_var) * eps
        return z


class SampleIWAELayer(lasagne.layers.MergeLayer):
    """
    importance sampling in Variational methods as desribed in [BURDA].xw

    Parameters
    ----------
    mu, log_var : class:`Layer` instances
        Parametrizing the mean and log(variance) of the distribution to sample
        from as described in [BURDA]. The code assumes that these have the same
        number of dimensions

    Eq_samples: Int or T.scalar
        Number of Monte Carlo samples used to estimate the expectation over
        q(z|x) in eq. (8) in [BURDA]

    iwae_samples: Int or T.scalar
        Number of importance samples in the sum over k in eq. (8) in [BURDA]

    References:
        [BURDA] Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov.
        "Importance Weighted Autoencoders."
        arXiv preprint arXiv:1509.00519 (2015).
    """

    def __init__(self, mu, var, Eq_samples=1, iwae_samples=10,**kwargs):
        super(SampleIWAELayer, self).__init__([mu, var], **kwargs)

        self.E_qsamples = Eq_samples
        self.iwae_samples = iwae_samples

        self._srng = RandomStreams(
            lasagne.random.get_rng().randint(1, 2147462579))


    def get_output_shape_for(self, input_shapes):
        batch_size, num_latent = input_shapes[0]
        if isinstance(batch_size, int) and isinstance(self.iwae_samples, int) and isinstance(self.E_qsamples, int):
            out_dim = (batch_size*self.E_qsamples*self.iwae_samples, num_latent)
        else:
            out_dim = (None, num_latent)
        return out_dim

    def get_output_for(self, input, **kwargs):
        mu, log_var = input
        #mu, log_var, (bs, num_latent)
        batch_size, num_latent = mu.shape
        eps = self._srng.normal([batch_size, self.E_qsamples, self.iwae_samples, num_latent],
                                dtype=theano.config.floatX)

        #z.shape: (batch_size, self.ivae_samples, num_latent)
        z = mu.dimshuffle(0,'x','x',1) + \
            T.exp(0.5 * log_var.dimshuffle(0,'x','x',1)) * eps

        return z.reshape((-1,num_latent))
