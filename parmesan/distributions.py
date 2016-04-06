import math
import theano.tensor as T


c = - 0.5 * math.log(2*math.pi)


def log_normal(x, mean, std, eps=0.0):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance, at values x.
    Variance is parameterized as standard deviation.

        .. math:: \log p(x) = \log \mathcal{N}(x; \mu, \sigma^2I)
    
    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf.
    mean : Theano tensor
        Mean of the Gaussian distribution.
    std : Theano tensor
        Standard deviation of the diagonal covariance Gaussian.
    eps : float
        Small number added to standard deviation to avoid NaNs.

    Returns
    -------
    Theano tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.

    See also
    --------
    log_normal1 : using variance parameterization
    log_normal2 : using log variance parameterization
    """
    std += eps
    return c - T.log(T.abs_(std)) - (x - mean)**2 / (2 * std**2)


def log_normal1(x, mean, var, eps=0.0):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance, at values x.
    Variance is parameterized as variance rather than standard deviation.

        .. math:: \log p(x) = \log \mathcal{N}(x; \mu, \sigma^2I)
    
    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf.
    mean : Theano tensor
        Mean of the Gaussian distribution.
    var : Theano tensor
        Variance of the diagonal covariance Gaussian.
    eps : float
        Small number added to variance to avoid NaNs.

    Returns
    -------
    Theano tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.

    See also
    --------
    log_normal : using standard deviation parameterization
    log_normal2 : using log variance parameterization
    """
    var += eps
    return c - T.log(var)/2 - (x - mean)**2 / (2*var)


def log_normal2(x, mean, log_var, eps=0.0):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance, at values x.
    Variance is parameterized as log variance rather than standard deviation, which ensures :math:`\sigma > 0`.

        .. math:: \log p(x) = \log \mathcal{N}(x; \mu, \sigma^2I)
    
    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf.
    mean : Theano tensor
        Mean of the Gaussian distribution.
    log_var : Theano tensor
        Log variance of the diagonal covariance Gaussian.
    eps : float
        Small number added to denominator to avoid NaNs.

    Returns
    -------
    Theano tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.

    See also
    --------
    log_normal : using standard deviation parameterization
    log_normal1 : using variance parameterization
    """
    return c - log_var/2 - (x - mean)**2 / (2 * T.exp(log_var) + eps)


def log_stdnormal(x):
    """
    Compute log pdf of a standard Gaussian distribution with zero mean and unit variance, at values x.

        .. math:: \log p(x) = \log \mathcal{N}(x; 0, I)
    
    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf.

    Returns
    -------
    Theano tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    """
    return c - x**2 / 2


def log_bernoulli(x, p, eps=0.0):
    """
    Compute log pdf of a Bernoulli distribution with success probability p, at values x.

        .. math:: \log p(x; p) = \log \mathcal{B}(x; p)

    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf.
    p : Theano tensor
        Success probability :math:`p(x=1)`, which is also the mean of the Bernoulli distribution.
    eps : float
        Small number used to avoid NaNs by clipping p in range [eps;1-eps].

    Returns
    -------
    Theano tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    """
    p = T.clip(p, eps, 1.0 - eps)
    return -T.nnet.binary_crossentropy(p, x)


def log_multinomial(x, p, eps=0.0):
    """
    Compute log pdf of multinomial distribution

        .. math:: \log p(x; p) = \sum_x p(x) \log q(x)

    where p is the true class probability and q is the predicted class
    probability.


    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf. Either an integer vector or a
        samples by class matrix with class probabilities.
    p : Theano tensor
        Samples by class matrix with predicted class probabilities.
    eps : float
        Small number used to avoid NaNs by offsetting p.

    Returns
    -------
    Theano tensor
        Element-wise log probability.
    """
    p += eps
    return -T.nnet.categorical_crossentropy(p, x)


def kl_normal1_stdnormal(mean, var, eps=0.0):
    """
    Closed-form solution of the KL-divergence between a Gaussian parameterized 
    with diagonal variance and a standard Gaussian.

    .. math::

        D_{KL}[\mathcal{N}(\mu, \sigma^2 I) || \mathcal{N}(0, I)]

    Parameters
    ----------
    mean : Theano tensor
        Mean of the diagonal covariance Gaussian.
    var : Theano tensor
        Variance of the diagonal covariance Gaussian.
    eps : float
        Small number added to variance to avoid NaNs.

    Returns
    -------
    Theano tensor
        Element-wise KL-divergence, this has to be summed when the Gaussian distributions are multi-variate.
    
    See also
    --------
    kl_normal2_stdnormal : using log variance parameterization
    """
    var += eps
    return -0.5*(1 + T.log(var) - mean**2 - var)

def kl_normal2_stdnormal(mean, log_var):
    """
    Compute closed-form solution to the KL-divergence between a Gaussian parameterized 
    with diagonal log variance and a standard Gaussian.

    In the setting of the variational autoencoder, when a Gaussian prior and diagonal Gaussian 
    approximate posterior is used, this analytically integrated KL-divergence term yields a lower variance 
    estimate of the likelihood lower bound compared to computing the term by Monte Carlo approximation.

        .. math:: D_{KL}[q_{\phi}(z|x) || p_{\theta}(z)]

    See appendix B of [KINGMA]_ for details.

    Parameters
    ----------
    mean : Theano tensor
        Mean of the diagonal covariance Gaussian.
    log_var : Theano tensor
        Log variance of the diagonal covariance Gaussian.

    Returns
    -------
    Theano tensor
        Element-wise KL-divergence, this has to be summed when the Gaussian distributions are multi-variate.

    See also
    --------
    kl_normal1_stdnormal : using variance parameterization

    References
    ----------
        ..  [KINGMA] Kingma, Diederik P., and Max Welling.
            "Auto-Encoding Variational Bayes."
            arXiv preprint arXiv:1312.6114 (2013).
    """
    return -0.5*(1 + log_var - mean**2 - T.exp(log_var))


def kl_normal1_normal1(mean1, var1, mean2, var2, eps=0.0):
    """
    Compute closed-form solution to the KL-divergence between two Gaussians parameterized 
    with diagonal variance.

    Parameters
    ----------
    mean1 : Theano tensor
        Mean of the q Gaussian.
    var1 : Theano tensor
        Variance of the q Gaussian.
    mean2 : Theano tensor
        Mean of the p Gaussian.
    var2 : Theano tensor
        Variance of the p Gaussian.
    eps : float
        Small number added to variances to avoid NaNs.

    Returns
    -------
    Theano tensor
        Element-wise KL-divergence, this has to be summed when the Gaussian distributions are multi-variate.

    See also
    --------
    kl_normal2_normal2 : using log variance parameterization
    """
    var1 += eps
    var2 += eps
    return 0.5*T.log(var2/var1) + (var1 + (mean1 - mean2)**2) / (2*var2) - 0.5

def kl_normal2_normal2(mean1, log_var1, mean2, log_var2, eps=0.0):
    """
    Compute closed-form solution to the KL-divergence between two Gaussians parameterized 
    with diagonal log variance.

    .. math::

       D_{KL}[q||p] &= -\int p(x) \log q(x) dx + \int p(x) \log p(x) dx     \\
                    &= -\int \mathcal{N}(x; \mu_2, \sigma^2_2) \log \mathcal{N}(x; \mu_1, \sigma^2_1) dx
                        + \int \mathcal{N}(x; \mu_2, \sigma^2_2) \log \mathcal{N}(x; \mu_2, \sigma^2_2) dx     \\
                    &= \frac{1}{2} \log(2\pi\sigma^2_2) + \frac{\sigma^2_1 + (\mu_1 - \mu_2)^2}{2\sigma^2_2} 
                        - \frac{1}{2}( 1 + \log(2\pi\sigma^2_1) )      \\
                    &= \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma^2_1 + (\mu_1 - \mu_2)^2}{2\sigma^2_2} - \frac{1}{2}

    Parameters
    ----------
    mean1 : Theano tensor
        Mean of the q Gaussian.
    log_var1 : Theano tensor
        Log variance of the q Gaussian.
    mean2 : Theano tensor
        Mean of the p Gaussian.
    log_var2 : Theano tensor
        Log variance of the p Gaussian.
    eps : float
        Small number added to denominator to avoid NaNs.

    Returns
    -------
    Theano tensor
        Element-wise KL-divergence, this has to be summed when the Gaussian distributions are multi-variate.

    See also
    --------
    kl_normal1_normal1 : using variance parameterization
    """
    return 0.5*log_var2 - 0.5*log_var1 + (T.exp(log_var1) + (mean1 - mean2)**2) / (2*T.exp(log_var2) + eps) - 0.5
