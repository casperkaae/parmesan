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

    seed : int
        seed to random stream

    Methods
    ----------
    seed : Helper function to change the random seed after init is called

    References
    ----------
        ..  [KINGMA] Kingma, Diederik P., and Max Welling.
            "Auto-Encoding Variational Bayes."
            arXiv preprint arXiv:1312.6114 (2013).
    """
    def __init__(self, mean, log_var,
                 seed=lasagne.random.get_rng().randint(1, 2147462579),
                 **kwargs):
        super(SimpleSampleLayer, self).__init__([mean, log_var], **kwargs)

        self._srng = RandomStreams(seed)

    def seed(self, seed=lasagne.random.get_rng().randint(1, 2147462579)):
       self._srng.seed(seed)

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
    mu : class:`Layer` instance
        Parameterizing the mean of the distribution to sample
        from as described in [BURDA]_.

    log_var : class:`Layer` instance
        By default assumed to parametrize log(sigma^2) of the distribution to
        sample from as described in [BURDA]_ which is transformed to sigma using
        the nonlinearity function as described below. Effectively this means
        that the nonlinearity function controls what log_var parametrizes. A few
        common examples:
        -nonlinearity = lambda x: T.exp(0.5*x) => log_var = log(sigma^2)[default]
        -nonlinearity = lambda x: T.sqrt(x) => log_var = sigma^2
        -nonlinearity = lambda x: x => log_var = sigma

    eq_samples : int or T.scalar
        Number of Monte Carlo samples used to estimate the expectation over
        q(z|x) in eq. (8) in [BURDA]_.

    iw_samples : int or T.scalar
        Number of importance samples in the sum over k in eq. (8) in [BURDA]_.

    nonlinearity : callable or None
        The nonlinearity that is applied to the log_var input layer to transform
        it into a standard deviation. By default we assume that
        log_var = log(sigma^2) and hence the corresponding nonlinearity is
        f(x) = T.exp(0.5*x) such that T.exp(0.5*log(sigma^2)) = sigma

    seed : int
        seed to random stream

    Methods
    ----------
    seed : Helper function to change the random seed after init is called

    References
    ----------
        ..  [BURDA] Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov.
            "Importance Weighted Autoencoders."
            arXiv preprint arXiv:1509.00519 (2015).
    """

    def __init__(self, mean, log_var,
                 eq_samples=1,
                 iw_samples=1,
                 nonlinearity=lambda x: T.exp(0.5*x),
                 seed=lasagne.random.get_rng().randint(1, 2147462579),
                  **kwargs):
        super(SampleLayer, self).__init__([mean, log_var], **kwargs)

        self.eq_samples = eq_samples
        self.iw_samples = iw_samples
        self.nonlinearity = nonlinearity

        self._srng = RandomStreams(seed)

    def seed(self, seed=lasagne.random.get_rng().randint(1, 2147462579)):
        self._srng.seed(seed)

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
            self.nonlinearity( log_var.dimshuffle(0,'x','x',1)) * eps

        return z.reshape((-1,num_latent))


class SimpleBernoulliSampleLayer(lasagne.layers.Layer):
    """
    Simple sampling layer drawing samples from bernoulli distributions.

    Parameters
    ----------
    mean : :class:`Layer` instances
          Parameterizing the mean value of each bernoulli distribution
    seed : int
        seed to random stream
    Methods
    ----------
    seed : Helper function to change the random seed after init is called
    """

    def __init__(self, mean,
                 seed=lasagne.random.get_rng().randint(1, 2147462579),
                 **kwargs):
        super(SimpleBernoulliSampleLayer, self).__init__(mean, **kwargs)

        self._srng = RandomStreams(seed)

    def seed(self, seed=lasagne.random.get_rng().randint(1, 2147462579)):
        self._srng.seed(seed)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, mu, **kwargs):
        return self._srng.binomial(size=mu.shape, p=mu, dtype=mu.dtype)


class BernoulliSampleLayer(lasagne.layers.Layer):
    """
    Bernoulli Sampling layer supporting importance sampling
    Parameters
    ----------
    mean : class:`Layer` instance
           Parameterizing the mean value of each bernoulli distribution
    eq_samples : int or T.scalar
        Number of Monte Carlo samples used to estimate the expectation over
    iw_samples : int or T.scalar
        Number of importance samples in the sum over k
    seed : int
        seed to random stream
    Methods
    ----------
    seed : Helper function to change the random seed after init is called
    """

    def __init__(self, mean,
                 eq_samples=1,
                 iw_samples=1,
                 seed=lasagne.random.get_rng().randint(1, 2147462579),
                  **kwargs):
        super(BernoulliSampleLayer, self).__init__(mean, **kwargs)

        self.eq_samples = eq_samples
        self.iw_samples = iw_samples

        self._srng = RandomStreams(seed)

    def seed(self, seed=lasagne.random.get_rng().randint(1, 2147462579)):
        self._srng.seed(seed)

    def get_output_shape_for(self, input_shape):
        batch_size, num_latent = input_shape
        if isinstance(batch_size, int) and \
           isinstance(self.iw_samples, int) and \
           isinstance(self.eq_samples, int):
            out_dim = (batch_size*self.eq_samples*self.iw_samples, num_latent)
        else:
            out_dim = (None, num_latent)
        return out_dim

    def get_output_for(self, input, **kwargs):
        mu = input
        batch_size, num_latent = mu.shape
        shp = (batch_size, self.eq_samples, self.iw_samples, num_latent)
        mu_shp = mu.dimshuffle(0,'x','x',1)
        mu_shp = T.repeat(mu_shp, axis=1, repeats=self.eq_samples)
        mu_shp = T.repeat(mu_shp, axis=2, repeats=self.iw_samples)
        samples = self._srng.binomial(
            size=shp, p=mu_shp, dtype=theano.config.floatX)
        return samples.reshape((-1, num_latent))
