# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

from sklearn.base import BaseEstimator


class DensityRatioMixin:
    def fit(self, X=None, y=None, numerator=None,
            denominator=None, n_samples=None, **kwargs):
        return self

    def predict(self, X, log=False, **kwargs):
        raise NotImplementedError

    def score(self, X, y, **kwargs):
        raise NotImplementedError


class InverseRatio(DensityRatioMixin, BaseEstimator):
    def __init__(self, base_ratio):
        self.base_ratio = base_ratio

    def fit(self, X=None, y=None, numerator=None,
            denominator=None, n_samples=None, **kwargs):
        raise NotImplementedError

    def predict(self, X, log=False, **kwargs):
        if log:
            return -self.base_ratio.predict(X, log=True, **kwargs)
        else:
            return 1. / self.base_ratio.predict(X, log=False, **kwargs)

    def score(self, X, y, **kwargs):
        raise NotImplementedError
