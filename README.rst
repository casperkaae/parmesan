Parmesan
=======
Parmesan is a library adding variational and semi-supervised neural network models to the neural network library `Lasagne
<http://github.com/Lasagne/Lasagne>`_.

Installation
------------
Parmesan depends heavily on the `Lasagne
<http://github.com/Lasagne/Lasagne>`_ and
`Theano
<http://deeplearning.net/software/theano>`_ libraries. Please make sure you have these installed before installing Parmesan.

**Install Parmesan**

.. code-block:: bash

  git clone https://github.com/casperkaae/parmesan.git
  cd parmesan
  python setup.py develop


Documentation
-------------
Work in progress. At the moment Parmesan primarily includes

* Layers for Monte Carlo approximation of integrals used in (importance weighted) variational autoencoders in *parmesan/layers/sample.py*
* Layers for constructing Ladder Networks in *parmesan/layers/ladderlayers.py*
* Layers for implementing normalizing flows in *parmesan/layers/flow.py*

Please see the source code and code examples for further details.

Examples
-------------
* **examples/vae_vanilla.py**: Variational autoencoder as described in Kingma. et. al. 2013
* **examples/iw_vae.py**: Variational autoencoder using importance sampling as described in Burda. et. al. 2015
* **examples/iw_vae_normflow.py**: Variational autoencoder using normalizing flows and importance sampling as described in Burda. et. al. 2015 and Rezende et al. 2015
* **examples/mnist_ladder.py**: Semi-supervised Ladder Network as described in Rasmus et. al. 2015


Development
-----------
Parmesan is work in progress, inputs, contributions and bug reports are very welcome.

The library was developed by
    * Casper Kaae Sønderby
    * Søren Kaae Sønderby
    * Lars Maaløe

References
-----------

* Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
* Burda, Y., Grosse, R., & Salakhutdinov, R. (2015). Importance Weighted Autoencoders. arXiv preprint arXiv:1509.00519.
* Rezende, D. J., & Mohamed, S. (2015). Variational Inference with Normalizing Flows. arXiv preprint arXiv:1505.05770.
* Rasmus, A., Valpola, H., Honkala, M., Berglund, M., & Raiko, T. (2015). Semi-Supervised Learning with Ladder Network. arXiv preprint arXiv:1507.02672.

