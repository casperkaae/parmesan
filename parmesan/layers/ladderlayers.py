import lasagne
import theano.tensor as T
from lasagne.layers import MergeLayer
from lasagne import init
from lasagne import nonlinearities


class DecoderNormalizeLayer(lasagne.layers.MergeLayer):
    """
        Special purpose layer used to construct the ladder network

        See the ladder_network example.
    """
    def __init__(self, incoming, mean, var, **kwargs):
        super(DecoderNormalizeLayer, self).__init__(
            [incoming, mean, var], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        input, mean, var = inputs
        return (input - mean) / T.sqrt(var)


class DenoiseLayer(MergeLayer):
    """
        Special purpose layer used to construct the ladder network

        See the ladder_network example.
    """
    def __init__(self, u_net, z_net,
                 nonlinearity=nonlinearities.sigmoid, **kwargs):
        super(DenoiseLayer, self).__init__([u_net, z_net], **kwargs)

        u_shp, z_shp = self.input_shapes

        if not len(u_shp) == 2 and len(z_shp) == 2:
            raise ValueError('Both u and z must be 2d. u was %s, z was %s' % (
                str(u_shp), str(z_shp)))

        if not u_shp[1] == z_shp[1]:
            raise ValueError("u and z must both be (num_batch, num_units)"
                             " u was %s, z was %s" % (str(u_shp), str(z_shp)))
        self.num_inputs = z_shp[1]
        self.nonlinearity = nonlinearity
        constant = init.Constant
        self.a1 = self.add_param(constant(0.), (self.num_inputs,), name="a1")
        self.a2 = self.add_param(constant(1.), (self.num_inputs,), name="a2")
        self.a3 = self.add_param(constant(0.), (self.num_inputs,), name="a3")
        self.a4 = self.add_param(constant(0.), (self.num_inputs,), name="a4")

        self.c1 = self.add_param(constant(0.), (self.num_inputs,), name="c1")
        self.c2 = self.add_param(constant(1.), (self.num_inputs,), name="c2")
        self.c3 = self.add_param(constant(0.), (self.num_inputs,), name="c3")

        self.c4 = self.add_param(constant(0.), (self.num_inputs,), name="c4")

        self.b1 = self.add_param(constant(0.), (self.num_inputs,),
                                 name="b1", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        output_shape = list(input_shapes[0])  # make a mutable copy
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        u, z_lat = inputs
        sigval = self.c1 + self.c2*z_lat
        sigval += self.c3*u + self.c4*z_lat*u
        sigval = self.nonlinearity(sigval)
        z_est = self.a1 + self.a2 * z_lat + self.b1*sigval
        z_est += self.a3*u + self.a4*z_lat*u

        return z_est
