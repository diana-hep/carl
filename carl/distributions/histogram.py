# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from itertools import product
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from scipy.interpolate import interp1d
from scipy.signal import medfilt

from .base import DistributionMixin


class Histogram(DistributionMixin):
    def __init__(self, bins=10, range=None, smoothing=None, interpolation=None,
                 random_state=None):
        super(Histogram, self).__init__(random_state=random_state)

        self.bins = bins
        self.range = range
        self.smoothing = smoothing
        self.interpolation = interpolation

    def pdf(self, X, **kwargs):
        X = check_array(X)

        if self.ndim_ == 1 and self.interpolation:
            return self.interpolation_(X[:, 0])

        all_indices = []

        for j in range(X.shape[1]):
            indices = np.searchsorted(self.edges_[j],
                                      X[:, j],
                                      side="right") - 1

            # For the last bin, the upper is inclusive
            indices[X[:, j] == self.edges_[j][-2]] -= 1
            all_indices.append(indices)

        return self.histogram_[all_indices]

    def nnlf(self, X, **kwargs):
        return -np.log(self.pdf(X, **kwargs))

    def rvs(self, n_samples, **kwargs):
        rng = check_random_state(self.random_state)

        # Draw random bins with respect to their densities
        h = (self.histogram_ / self.histogram_.sum()).ravel()
        flat_indices = np.searchsorted(np.cumsum(h),
                                       rng.rand(n_samples),
                                       side="right")

        # Build bin corners
        indices = np.unravel_index(flat_indices, self.histogram_.shape)
        indices_end = [a + 1 for a in indices]
        shape = [len(d) for d in self.edges_] + [len(self.edges_)]
        corners = np.array(list(product(*self.edges_))).reshape(shape)

        # Draw uniformly within bins
        low = corners[indices]
        high = corners[indices_end]
        u = rng.rand(*low.shape)

        return low + u * (high - low)

    def fit(self, X, y=None, sample_weight=None, **kwargs):
        # Checks
        X = check_array(X)

        if sample_weight is not None and len(sample_weight) != len(X):
            raise ValueError

        # Compute histogram and edges
        h, e = np.histogramdd(X, bins=self.bins, range=self.range,
                              weights=sample_weight, normed=True)

        # Add empty bins for out of bound samples
        for j in range(X.shape[1]):
            h = np.insert(h, 0, 0., axis=j)
            h = np.insert(h, h.shape[j], 0., axis=j)
            e[j] = np.insert(e[j], 0, -np.inf)
            e[j] = np.insert(e[j], len(e[j]), np.inf)

        if X.shape[1] == 1:
            if self.smoothing == "353qh":
                h[1:-1] = _353qh(h[1:-1])
            elif self.smoothing == "353qh_twice":
                h[1:-1] = _353qh_twice(h[1:-1])

            if self.interpolation:
                inputs = e[0][2:-1] - (e[0][2] - e[0][1]) / 2.
                outputs = h[1:-1]
                self.interpolation_ = interp1d(inputs, outputs,
                                               kind=self.interpolation,
                                               bounds_error=False,
                                               fill_value=0.)

        self.histogram_ = h
        self.edges_ = e
        self.ndim_ = X.shape[1]

        return self

    def score(self, X, y=None, **kwargs):
        raise NotImplementedError

    @property
    def ndim(self):
        return self.ndim_


def _median3(y):
    z = y.copy()
    z[0] = np.median([y[0], y[1], 3*y[1] - 2*y[2]])
    z[-1] = np.median([y[-1], y[-2], 3*y[-2] - 2*y[-3]])
    z[1:-1] = medfilt(y, 3)[1:-1]
    return z


def _median5(y):
    z = y.copy()
    z[1] = np.median(y[0:3])
    z[-2] = np.median(y[-3:])
    z[2:-2] = medfilt(y, 5)[2:-2]
    return z


def _quadratic(y):
    z = y.copy()

    for i in range(2, len(y) - 2):
        if y[i - 1] != y[i]:
            continue
        if y[i] != y[i + 1]:
            continue
        h0 = y[i - 2] - y[i]
        h1 = y[i + 2] - y[i]
        if h0 * h1 <= 0:
            continue
        j = 1
        if abs(h1) > abs(h0):
            j -= 1
        z[i] = -0.5 * y[i - 2*j] + y[i] / 0.75 + y[i + 2*j] / 6
        z[i+j] = 0.5 * (y[i + 2*j] - y[i - 2*j]) + y[i]

    return z


def _353qh(y):
    # 353 step
    z = _median3(y)
    z = _median5(z)
    z[1:-1] = _median3(z)[1:-1]

    # quadratic step
    z = _quadratic(z)

    # hanning step
    z[1:-1] = 0.25 * z[:-2] + 0.5 * z[1:-1] + 0.25 * z[2:]

    return z


def _353qh_twice(y):
    z = _353qh(y)
    r = _353qh(y - z)
    return z + r
