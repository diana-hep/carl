# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.utils import check_random_state

from .base import DistributionMixin


class Sampler(DistributionMixin):
    def __init__(self, random_state=None):
        super(Sampler, self).__init__(random_state=random_state)

    def fit(self, X, y=None, sample_weight=None, **kwargs):
        self.X_ = X
        self.ndim_ = X.shape[1]
        self.sample_weight_ = sample_weight
        return self

    def rvs(self, n_samples, **kwargs):
        random_state = check_random_state(self.random_state)

        if self.sample_weight_ is None:
            w = np.ones(len(self.X_))
        else:
            w = self.sample_weight_

        w = w / w.sum()
        indices = np.searchsorted(np.cumsum(w), random_state.rand(n_samples))

        return self.X_[indices]

    def ndim(self, **kwargs):
        return self.ndim_
