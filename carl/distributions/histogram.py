# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


class Histogram(BaseEstimator):
    def __init__(self, random_state=None):
        # XXX build a wrapper around np.histogramdd
        self.random_state = random_state

    def pdf(self, X):
        raise NotImplementedError

    def nnlf(self, X):
        raise NotImplementedError

    def cdf(self, X):
        raise NotImplementedError

    def rvs(self, n_samples):
        raise NotImplementedError

    def fit(self, X, y=None):
        return self

    def score(self, X, y=None):
        raise NotImplementedError
