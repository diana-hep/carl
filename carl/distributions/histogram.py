# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import astropy.stats as astropy


from itertools import product
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from scipy.interpolate import interp1d

from .base import DistributionMixin


class Histogram(DistributionMixin):
    """N-dimensional histogram."""

    def __init__(self, bins=10, range=None, interpolation=None, var_width=False):
        """Constructor.

        Parameters
        ----------
        * `bins` [int]:
            The number of bins.

        * `range` [list of bounds (low, high)]:
            The boundaries. If `None`, bounds are inferred from data.

        * `interpolation` [string, optional]
            Specifies the kind of interpolation between bins as a string
            (`"linear"`, `"nearest"`, `"zero"`, `"slinear"`, `"quadratic"`,
            `"cubic"`).
        """
        self.bins = bins
        self.range = range
        self.interpolation = interpolation
        self.var_width = var_width

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

    def nll(self, X, **kwargs):
        return -np.log(self.pdf(X, **kwargs))

    def rvs(self, n_samples, random_state=None, **kwargs):
        rng = check_random_state(random_state)

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

    def variable_density(self, h, bins):
        widths = bins[1:] - bins[:-1]
        norm_h = h / widths / h.sum()
        return norm_h
    
    def fit(self, X, sample_weight=None, **kwargs):
        # Checks
        X = check_array(X)
        if sample_weight is not None and len(sample_weight) != len(X):
            raise ValueError
        # Compute histogram and edges
        if self.bins == 'blocks':
            bins = astropy.bayesian_blocks(X.ravel(), fitness='events',p0=0.0001)
            #da, bins = astropy.scott_bin_width(X.ravel(), True)
            h, e = np.histogram(X.ravel(), bins=bins, range=self.range[0],
                                  weights=sample_weight,normed=False)

            e = [e]
        elif self.var_width:
            splits = np.array_split(np.sort(X.ravel()), self.bins)
            base = np.array([splits[0][0]] + [x[-1] for x in splits])
            h, e = np.histogram(X.ravel(), bins=base, range=self.range[0],normed=False, 
                                weights=sample_weight)
            h = self.variable_density(h,e)
            e = [e]
           
        else:
            bins = self.bins
            h, e = np.histogramdd(X, bins=bins, range=self.range,
                                  weights=sample_weight,normed=False) 
        
        # Add empty bins for out of bound samples
        for j in range(X.shape[1]):
            h = np.insert(h, 0, 0., axis=j)
            h = np.insert(h, h.shape[j], 0., axis=j)
            e[j] = np.insert(e[j], 0, -np.inf)
            e[j] = np.insert(e[j], len(e[j]), np.inf)

        if X.shape[1] == 1 and self.interpolation:
            inputs = e[0][2:-1] - (e[0][2] - e[0][1]) / 2.
            inputs[0] = e[0][1]
            inputs[-1] = e[0][-2]
            outputs = h[1:-1]
            self.interpolation_ = interp1d(inputs, outputs,
                                           kind=self.interpolation,
                                           bounds_error=False,
                                           fill_value=0.)

        self.histogram_ = h
        self.edges_ = e
        self.ndim_ = X.shape[1]

        return self

    def cdf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    def ppf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    @property
    def ndim(self):
        return self.ndim_
