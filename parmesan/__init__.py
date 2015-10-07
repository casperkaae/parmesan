"""
Tools to train variational neural nets in Theano
"""

try:
    import theano
except ImportError:  # pragma: no cover
    raise ImportError("""Could not import Theano.

Please make sure you install a recent enough version of Theano.  See
section 'Install from PyPI' in the installation docs for more details:
http://parmasan.readthedocs.org/en/latest/user/installation.html#install-from-pypi
""")
else:
    del theano


from . import distributions
from . import datasets
from . import layers
from . import utils
from . import preprocessing

import pkg_resources
__version__ = pkg_resources.get_distribution("parmesan").version
del pkg_resources
