import lasagne
import theano.tensor as T
import theano
import numpy as np


class NormalizeLayer(lasagne.layers.Layer):

    def __init__(self, incoming, axes=None, epsilon=1e-10, alpha='single_pass',
                 return_stats=False, stat_indices=None,
                 **kwargs):
        """
        This layer is a modified version of code originally written by
        Jan Schluter.

        Instantiates a layer performing batch normalization of its inputs [1]_.

        Params
        ------
        incoming: `Layer` instance or expected input shape

        axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)

        epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems

        alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
            If alpha is none we'll assume that the entire training set
            is passed through in one batch.

        return_stats: return mean and std

        stat_indices if slice object only calc stats for these indices. Used
            semisupervsed learning


        Notes
        -----
        This layer accepts the keyword collect=True when get_output is
        called. Before evaluation you should collect the batchnormalizatino
        statistics by running all you data through a function with

        collect=True and deterministic=True

        Then you can evaluate.



        References
        ----------
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization:
               Accelerating deep network training by reducing internal
               covariate shift."
               arXiv preprint arXiv:1502.03167 (2015).

        """
        super(NormalizeLayer, self).__init__(incoming, **kwargs)
        self.return_stats = return_stats
        self.stat_indices = stat_indices
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("NormalizeLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        self.mean = self.add_param(lasagne.init.Constant(0), shape, 'mean',
                                   trainable=False, regularizable=False)
        self.var = self.add_param(lasagne.init.Constant(1), shape, 'var',
                                  trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False, collect=False,
                       **kwargs):

        if collect:
            # use this batch's mean and var
            if self.stat_indices is None:
                mean = input.mean(self.axes, keepdims=True)
                var = input.var(self.axes, keepdims=True)
            else:
                mean = input[self.stat_indices].mean(self.axes, keepdims=True)
                var = input[self.stat_indices].var(self.axes, keepdims=True)
            # and update the stored mean and var:
            # we create (memory-aliased) clones of the stored mean and var
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_var = theano.clone(self.var, share_inputs=False)
            # set a default update for them

            if self.alpha is not 'single_pass':
                running_mean.default_update = (
                    (1 - self.alpha) * running_mean + self.alpha * mean)
                running_var.default_update = (
                    (1 - self.alpha) * running_var + self.alpha * var)
            else:
                print "Collecting using single pass..."
                # this is ugly figure out what can be safely removed...
                running_mean.default_update = (0 * running_mean + 1.0 * mean)
                running_var.default_update = (0 * running_var + 1.0 * var)

            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later)
            mean += 0 * running_mean
            var += 0 * running_var

        elif deterministic:
            # use stored mean and var
            mean = self.mean
            var = self.var
        else:
            # use this batch's mean and var
            mean = input.mean(self.axes, keepdims=True)
            var = input.var(self.axes, keepdims=True)

        mean = T.addbroadcast(mean, *self.axes)
        var = T.addbroadcast(var, *self.axes)
        normalized = (input - mean) / T.sqrt(var + self.epsilon)

        if self.return_stats:
            return [normalized, mean, var]
        else:
            return normalized


class ScaleAndShiftLayer(lasagne.layers.Layer):
    """
    This layer is a modified version of code originally written by
    Jan Schluter.

    Used with the NormalizeLayer to construct a batchnormalization layer.

    Params
    ------
    incoming: `Layer` instance or expected input shape

    axes: int or tuple of int denoting the axes to normalize over;
        defaults to all axes except for the second if omitted (this will
        do the correct thing for dense layers and convolutional layers)
    """

    def __init__(self, incoming, axes=None, **kwargs):
        super(ScaleAndShiftLayer, self).__init__(incoming, **kwargs)
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        self.beta = self.add_param(lasagne.init.Constant(0), shape, name='beta',
                                   trainable=True, regularizable=True)
        self.gamma = self.add_param(lasagne.init.Constant(1), shape, name='gamma',
                                    trainable=True, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        beta = T.addbroadcast(self.beta, *self.axes)
        gamma = T.addbroadcast(self.gamma, *self.axes)
        return input*gamma + beta
