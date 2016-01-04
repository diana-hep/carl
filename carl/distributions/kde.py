# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from scipy.stats import gaussian_kde
from sklearn.utils import check_random_state
from sklearn.utils import check_array

from .base import DistributionMixin


class KernelDensity(DistributionMixin):
    def __init__(self, bandwidth=None, random_state=None):
        super(KernelDensity, self).__init__(random_state=random_state)
        self.bandwidth = bandwidth

    def pdf(self, X, **kwargs):
        X = check_array(X)
        return self.kde_.pdf(X.T)

    def nnlf(self, X, **kwargs):
        X = check_array(X)
        return -self.kde_.logpdf(X.T)

    def rvs(self, n_samples, **kwargs):
        # gaussian_kde uses Numpy global random state...
        return self.kde_.resample(n_samples).T

    def fit(self, X, y=None, **kwargs):
        X = check_array(X).T
        self.kde_ = gaussian_kde(X, bw_method=self.bandwidth)
        return self

    def score(self, X, y=None, **kwargs):
        X = check_array(X)
        return self.kde_.logpdf(X.T).sum()
