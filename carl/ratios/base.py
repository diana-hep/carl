# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.metrics import mean_squared_error


class DensityRatioMixin:
    def fit(self, X=None, y=None, numerator=None,
            denominator=None, n_samples=None, **kwargs):
        return self

    def predict(self, X, log=False, **kwargs):
        raise NotImplementedError

    def score(self, X, y, finite_only=True, **kwargs):
        ratios = self.predict(X, **kwargs)

        if finite_only:
            mask = np.isfinite(ratios)
            y = y[mask]
            ratios = ratios[mask]

        return -mean_squared_error(y, ratios)


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


class DecomposedRatio(DensityRatioMixin, BaseEstimator):
    def __init__(self, base_ratio):
        self.base_ratio = base_ratio

    def fit(self, X=None, y=None, numerator=None,
            denominator=None, n_samples=None, **kwargs):
        self.identity_ = (numerator is not None) and (numerator is denominator)

        if self.identity_:
            return self

        if numerator is None or denominator is None or n_samples is None:
            raise ValueError

        self.ratios_ = {}
        self.ratios_map_ = {}
        self.numerator_ = numerator
        self.denominator_ = denominator

        n_samples_ij = n_samples // (len(numerator.components) *
                                     len(denominator.components))

        # XXX in case of identities or inverses, samples are thrown away!
        #     we should first check whether these cases exist, and then
        #     assign n_samples to each sub-ratio

        for i, p_i in enumerate(numerator.components):
            for j, p_j in enumerate(denominator.components):
                if (p_i, p_j) in self.ratios_map_:
                    ratio = InverseRatio(
                        self.ratios_[self.ratios_map_[(p_i, p_j)]])

                else:
                    ratio = clone(self.base_ratio)
                    ratio.fit(numerator=p_j, denominator=p_i,
                              n_samples=n_samples_ij)

                self.ratios_[(j, i)] = ratio
                self.ratios_map_[(p_j, p_i)] = (j, i)

        return self

    def predict(self, X, log=False, **kwargs):
        if self.identity_:
            if log:
                return np.zeros(len(X))
            else:
                return np.ones(len(X))

        else:
            w_num = self.numerator_.compute_weights(**kwargs)
            w_den = self.denominator_.compute_weights(**kwargs)

            r = np.zeros(len(X))

            for i, w_i in enumerate(w_num):
                s = np.zeros(len(X))

                for j, w_j in enumerate(w_den):
                    s += w_j * self.ratios_[(j, i)].predict(X, **kwargs)

                r += w_i / s

            if log:
                return np.log(r)
            else:
                return r
