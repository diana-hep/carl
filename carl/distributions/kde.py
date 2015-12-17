# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.neighbors import KernelDensity as _KernelDensity
from sklearn.utils import check_random_state

from .base import DistributionMixin


class KernelDensity(DistributionMixin):
    def __init__(self, bandwidth=1.0, algorithm="auto",
                 kernel="gaussian", metric="euclidean", atol=0, rtol=0,
                 breadth_first=True, leaf_size=40, metric_params=None,
                 random_state=None):
        super(KernelDensity, self).__init__(random_state=random_state)

        self.algorithm = algorithm
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params

    def pdf(self, X, **kwargs):
        return np.exp(self.kde_.score_samples(X))

    def nnlf(self, X, **kwargs):
        return -self.kde_.score_samples(X)

    def rvs(self, n_samples, **kwargs):
        rng = check_random_state(self.random_state)
        return self.kde_.sample(n_samples=n_samples, random_state=rng)

    def fit(self, X, y=None, **kwargs):
        self.kde_ = _KernelDensity(bandwidth=self.bandwidth,
                                   algorithm=self.algorithm,
                                   kernel=self.kernel,
                                   metric=self.metric,
                                   atol=self.atol,
                                   rtol=self.rtol,
                                   breadth_first=self.breadth_first,
                                   leaf_size=self.leaf_size,
                                   metric_params=self.metric_params)
        self.kde_.fit(X, y=y)
        return self

    def score(self, X, y=None, **kwargs):
        return self.kde_.score(X, y=y)
