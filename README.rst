Parmesan
=======
Parmesan is a library adding variational and semi-supervised neural network models to [Lasagne](https://github.com/Lasagne/Lasagne).

Installation
------------
Parmesan depends heavily on the [Lasagne](https://github.com/Lasagne/Lasagne) and [Theano](http://deeplearning.net/software/theano/) libraries. Make sure these are installed

Install using pip

.. code-block:: bash

  pip install https://github.com/casperkaae/parmesan.git --user --no-deps


Install from source

.. code-block:: bash

  git clone https://github.com/casperkaae/parmesan.git
  cd parmesan
  python setup.py develop


Documentation
-------------
Work in progress. Please see the examples folder for working code

Examples
-------------
* **examples/vae_vanilla.py**: Variational autoencoder as described in Kingma. et. al. 2013 "Autoencoding Variational Bayes"
* **examples/iw_vae.py**: Variational autoencoder using importance sampling as described in Burda. et. al. 2015 "Importance Weighted Autoencoders"
* **examples/mnist_ladder.py**: Semi-supervised Ladder Network as described in Rasmus et. al. 2015 "Semi-Supervised Learning with Ladder Network"


Development
-----------
Parmesan is work in progress, inputs and contributions are very welcome.

The library was developed by
    * Casper Kaae Sønderby
    * Søren Kaae Sønderby
    * Lars Maaløe