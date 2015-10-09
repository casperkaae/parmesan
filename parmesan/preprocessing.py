"""
This file contains code from https://github.com/arasmus/ladder/

Released with the license:

The MIT License (MIT)

Copyright (c) 2015 arasmus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np


class ContrastNorm(object):
    """
    Code copied from https://github.com/arasmus/ladder/blob/master/nn.py
    """
    def __init__(self, scale=55, epsilon=1e-8):
        raise Warning('ContranstNorm is untested!')
        self.scale = np.float32(scale)
        self.epsilon = np.float32(epsilon)

    def apply(self, data, copy=False):
        if copy:
            data = np.copy(data)
        data_shape = data.shape
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], np.product(data.shape[1:]))

        assert len(data.shape) == 2, 'Contrast norm on flattened data'

        data -= data.mean(axis=1)[:, np.newaxis]

        norms = np.sqrt(np.sum(data ** 2, axis=1)) / self.scale
        norms[norms < self.epsilon] = np.float32(1.)

        data /= norms[:, np.newaxis]

        if data_shape != data.shape:
            data = data.reshape(data_shape)

        return data


class ZCA(object):
    """
    Code copied from https://github.com/arasmus/ladder/blob/master/nn.py
    """
    def __init__(self, n_components=None, data=None, filter_bias=0.1):
        raise Warning('ZCA is untested!')
        self.filter_bias = np.float32(filter_bias)
        self.P = None
        self.P_inv = None
        self.n_components = 0
        self.is_fit = False
        if n_components and data:
            self.fit(n_components, data)

    def fit(self, n_components, data):
        try:
            import scipy
        except:
            raise ValueError('ZCA requires scipy')
        if len(data.shape) == 2:
            self.reshape = None
        else:
            assert n_components == np.product(data.shape[1:]), \
                'ZCA whitening components should be %d for convolutional data'\
                % np.product(data.shape[1:])
            self.reshape = data.shape[1:]

        data = self._flatten_data(data)
        assert len(data.shape) == 2
        n, m = data.shape
        self.mean = np.mean(data, axis=0)

        bias = self.filter_bias * scipy.sparse.identity(m, 'float32')
        cov = np.cov(data, rowvar=0, bias=1) + bias
        eigs, eigv = scipy.linalg.eigh(cov)

        assert not np.isnan(eigs).any()
        assert not np.isnan(eigv).any()
        assert eigs.min() > 0

        if self.n_components:
            eigs = eigs[-self.n_components:]
            eigv = eigv[:, -self.n_components:]

        sqrt_eigs = np.sqrt(eigs)
        self.P = np.dot(eigv * (1.0 / sqrt_eigs), eigv.T)
        assert not np.isnan(self.P).any()
        self.P_inv = np.dot(eigv * sqrt_eigs, eigv.T)

        self.P = np.float32(self.P)
        self.P_inv = np.float32(self.P_inv)

        self.is_fit = True

    def apply(self, data, remove_mean=True):
        data = self._flatten_data(data)
        d = data - self.mean if remove_mean else data
        return self._reshape_data(np.dot(d, self.P))

    def inv(self, data, add_mean=True):
        d = np.dot(self._flatten_data(data), self.P_inv)
        d += self.mean if add_mean else 0.
        return self._reshape_data(d)

    def _flatten_data(self, data):
        if self.reshape is None:
            return data
        assert data.shape[1:] == self.reshape
        return data.reshape(data.shape[0], np.product(data.shape[1:]))

    def _reshape_data(self, data):
        assert len(data.shape) == 2
        if self.reshape is None:
            return data
        return np.reshape(data, (data.shape[0],) + self.reshape)
