# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from theano.tensor.sharedvar import SharedVariable


class ParameterStacker(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y=None):
        mapping = {}

        for j, p in enumerate(sorted(self.kwargs.keys())):
            mapping[p] = j

        self.mapping_ = mapping

        return self

    def transform(self, X, y=None):
        Xp = np.empty((len(X), len(self.kwargs)))

        for p, v in self.kwargs.items():
            if isinstance(v, SharedVariable):
                Xp[:, self.mapping_[p]] = v.eval()

            else:
                Xp[:, self.mapping_[p]] = v

        return np.hstack((X, Xp))

    def get_params(self, deep=True):
        p = super(ParameterStacker, self).get_params(deep=deep)
        p.update(self.kwargs)
        return p

    def set_params(self, **params):
        for p, v in params.items():
            if p in self.kwargs:
                if isinstance(self.kwargs[p], SharedVariable):
                    self.kwargs[p].set_value(v)
                else:
                    self.kwargs[p] = v
            else:
                super(ParameterStacker, self).set_params(**{p: v})
