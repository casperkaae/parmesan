import numpy as np
import theano.tensor as T

from lasagne import init
from lasagne import nonlinearities

from lasagne.layers import Layer


class WeightNormalizationLayer(Layer):
    """
    A fully connected where the weight matrix is parameterized as in [1]_.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    v : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.

    g : Theano shared variable, expression, numpy array, callable
        Initial value, expression or initializer for the scaling. Biases should
        be a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.


    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.

    Based on Lasagne's Dense layer class.

    References
    ----------
    ..[1] Salimans, Tim, and Diederik P. Kingma. "Weight Normalization: A
          Simple Reparameterization to Accelerate Training of Deep Neural
          Networks." arXiv preprint arXiv:1602.07868 (2016).
    """
    def __init__(self, incoming, num_units, v=init.GlorotUniform(),
                 g=init.Constant(0.),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(WeightNormalizationLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.v = self.add_param(v, (num_inputs, num_units), name="v")
        self.g = self.add_param(g, (num_units,), name="g")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        v_norm = self.v.norm(2, axis=0).dimshuffle('x', 0)
        W = (self.v / v_norm) * self.g.dimshuffle('x', 0)
        activation = T.dot(input, W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


if __name__ == '__main__':
    from lasagne.layers import *
    l_in = InputLayer((None, 100))
    l_w = WeightNormalizationLayer(l_in, num_units=200)
    sym_x = T.matrix()
    np_x = np.random.random((16, 100)).astype('float32')
    output = get_output(l_w, sym_x)
    print output.eval({sym_x:np_x}).shape
