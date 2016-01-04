import lasagne
import theano.tensor as T
import numpy as np

class NormalizingPlanarFlowLayer(lasagne.layers.Layer):
    """
    Normalizing Planar Flow Layer as described in Rezende et
    al. [REZENDE]_ (Equation numbers and appendixes refers to this paper)
    Eq. (8) is used for calculating the forward transformation f(z).
    The last term of eq. (13) is also calculated within this layer and
    returned as an output for computational reasons. Furthermore, the
    transformation is ensured to be invertible using the constrains
    described in appendix A.1

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    u,w : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    References
    ----------
        ..  [REZENDE] Rezende, Danilo Jimenez, and Shakir Mohamed.
            "Variational Inference with Normalizing Flows."
            arXiv preprint arXiv:1505.05770 (2015).
    """
    def __init__(self, incoming, u=lasagne.init.Normal(),
                 w=lasagne.init.Normal(),
                 b=lasagne.init.Constant(0.0), **kwargs):
        super(NormalizingPlanarFlowLayer, self).__init__(incoming, **kwargs)
        
        num_latent = int(np.prod(self.input_shape[1:]))
        
        self.u = self.add_param(u, (num_latent,), name="u")
        self.w = self.add_param(w, (num_latent,), name="w")
        self.b = self.add_param(b, tuple(), name="b") # scalar
    
    def get_output_shape_for(self, input_shape):
        return input_shape
    
    
    def get_output_for(self, input, **kwargs):
        # 1) calculate u_hat to ensure invertibility (appendix A.1 to)
        # 2) calculate the forward transformation of the input f(z) (Eq. 8)
        # 3) calculate u_hat^T psi(z) 
        # 4) calculate logdet-jacobian log|1 + u_hat^T psi(z)| to be used in the LL function
        
        z = input
        # z is (batch_size, num_latent_units)
        uw = T.dot(self.u,self.w)
        muw = -1 + T.nnet.softplus(uw) # = -1 + T.log(1 + T.exp(uw))
        u_hat = self.u + (muw - uw) * T.transpose(self.w) / T.sum(self.w**2)
        zwb = T.dot(z,self.w) + self.b
        f_z = z + u_hat.dimshuffle('x',0) * lasagne.nonlinearities.tanh(zwb).dimshuffle(0,'x')
        
        psi = T.dot( (1-lasagne.nonlinearities.tanh(zwb)**2).dimshuffle(0,'x'),  self.w.dimshuffle('x',0)) # tanh(x)dx = 1 - tanh(x)**2
        psi_u = T.dot(psi, u_hat)

        logdet_jacobian = T.log(T.abs_(1 + psi_u))
        
        return [f_z, logdet_jacobian]
