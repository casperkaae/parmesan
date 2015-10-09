Parmesan
=======
Parmesan is a library adding variational and semi-supervised neural network models to the neural network library `Lasagne
<http://github.com/Lasagne/Lasagne>`_.

Installation
------------
Parmesan depends heavily on the `Lasagne
<http://github.com/Lasagne/Lasagne>`_ and
`Theano
<http://deeplearning.net/software/theano>`_. libraries.

.. code-block:: bash

  git clone https://github.com/casperkaae/parmesan.git
  cd parmesan
  python setup.py develop


Documentation
-------------
Work in progress. At the moment Parmesan primarily includes layers for
* monte carlo approximation of integrals in **parmesan/layers/sample.py**
* constructing Ladder Networks in **parmesan/layers/ladderlayers.py**

Please see these for further details. Further see the examples section below for working examples.

Examples
-------------
* **examples/vae_vanilla.py**: Variational autoencoder as described in Kingma. et. al. 2013
* **examples/iw_vae.py**: Variational autoencoder using importance sampling as described in Burda. et. al. 2015
* **examples/mnist_ladder.py**: Semi-supervised Ladder Network as described in Rasmus et. al.


Development
-----------
Parmesan is work in progress, inputs and contributions are very welcome.

The library was developed by
    * Casper Kaae Sønderby
    * Søren Kaae Sønderby
    * Lars Maaløe

References
-----------

* Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
* Burda, Y., Grosse, R., & Salakhutdinov, R. (2015). Importance Weighted Autoencoders. arXiv preprint arXiv:1509.00519.
* Rasmus, A., Valpola, H., Honkala, M., Berglund, M., & Raiko, T. (2015). Semi-Supervised Learning with Ladder Network. arXiv preprint arXiv:1507.02672.

