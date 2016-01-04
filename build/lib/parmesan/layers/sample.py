import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
import theano

class SimpleSampleLayer(lasagne.layers.MergeLayer):
    """
    Simple sampling layer drawing a single Monte Carlo sample to approximate
    E_q [log( p(x,z) / q(z|x) )]. This is the approach described in [KINGMA]_.

    Parameters
    ----------
    mu, log_var : :class:`Layer` instances
        Parameterizing the mean and log(variance) of the distribution to sample
        from as described in [KINGMA]_. The code assumes that these have the
        same number of dimensions.

    References
    ----------
        ..  [KINGMA] Kingma, Diederik P., and Max Welling.
            "Auto-Encoding Variational Bayes."
            arXiv preprint arXiv:1312.6114 (2013).
    """
    def __init__(self, mu, log_var, **kwargs):
        super(SimpleSampleLayer, self).__init__([mu, log_var], **kwargs)

        self._srng = RandomStreams(
            lasagne.random.get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        mu, log_var = input
        eps = self._srng.normal(mu.shape)
        z = mu + T.exp(0.5 * log_var) * eps
        return z


class SampleLayer(lasagne.layers.MergeLayer):
    """
    Sampling layer supporting importance sampling as described in [BURDA]_ and
    multiple Monte Carlo samples for the approximation of
    E_q [log( p(x,z) / q(z|x) )].

    Parameters
    ----------
    mu, log_var : :class:`Layer` instances
        Parameterizing the mean and log(variance) of the distribution to sample
        from as described in [BURDA]_. The code assumes that these have the same
        number of dimensions.

    eq_samples : int or T.scalar
        Number of Monte Carlo samples used to estimate the expectation over
        q(z|x) in eq. (8) in [BURDA]_.

    iw_samples : int or T.scalar
        Number of importance samples in the sum over k in eq. (8) in [BURDA]_.

    References
    ----------
        ..  [BURDA] Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov.
            "Importance Weighted Autoencoders."
            arXiv preprint arXiv:1509.00519 (2015).
    """

    def __init__(self, mu, log_var, eq_samples=1, iw_samples=1, **kwargs):
        super(SampleLayer, self).__init__([mu, log_var], **kwargs)

        self.eq_samples = eq_samples
        self.iw_samples = iw_samples

        self._srng = RandomStreams(
            lasagne.random.get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        batch_size, num_latent = input_shapes[0]
        if isinstance(batch_size, int) and \
           isinstance(self.iw_samples, int) and \
           isinstance(self.eq_samples, int):
            out_dim = (batch_size*self.eq_samples*self.iw_samples, num_latent)
        else:
            out_dim = (None, num_latent)
        return out_dim

    def get_output_for(self, input, **kwargs):
        mu, log_var = input
        batch_size, num_latent = mu.shape
        eps = self._srng.normal(
            [batch_size, self.eq_samples, self.iw_samples, num_latent],
             dtype=theano.config.floatX)

        z = mu.dimshuffle(0,'x','x',1) + \
            T.exp(0.5 * log_var.dimshuffle(0,'x','x',1)) * eps

        return z.reshape((-1,num_latent))
