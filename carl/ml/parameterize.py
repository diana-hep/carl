# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

from itertools import product
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone


class ParameterStacker(BaseEstimator, TransformerMixin):
    def __init__(self, params):
        self.params = params

    def transform(self, X, y=None):
        Xp = np.empty((len(X), len(self.params)))

        for i, p in enumerate(self.params):
            Xp[:, i] = p.eval()

        return np.hstack((X, Xp))


class _ParameterizedEstimator(BaseEstimator):
    def __init__(self, base_estimator, params):
        self.base_estimator = base_estimator
        self.params = params

    def _validate_X(self, X):
        if X.shape[1] == self.n_features_:
            X = self.stacker_.transform(X)
        assert X.shape[1] == self.n_features_ + len(self.params)

        return X

    def fit(self, X, y):
        self.stacker_ = ParameterStacker(self.params)

        # XXX: this assumes that X is extended with parameters
        self.n_features_ = X.shape[1] - len(self.params)
        self.estimator_ = clone(self.base_estimator).fit(X, y)

        return self

    def predict(self, X):
        return self.estimator_.predict(self._validate_X(X))


class ParameterizedClassifier(_ParameterizedEstimator, ClassifierMixin):
    def predict_proba(self, X):
        return self.estimator_.predict_proba(self._validate_X(X))


class ParameterizedRegressor(_ParameterizedEstimator, RegressorMixin):
    pass


def make_parameterized_classification(p0, p1, n_samples, params):
    if not isinstance(params[0], tuple):
        X0 = p0.rvs(n_samples // 2)
        X1 = p1.rvs(n_samples - (n_samples // 2))
        X = ParameterStacker(params).transform(np.vstack((X0, X1)))
        y = np.zeros(n_samples)
        y[len(X0):] = 1

        return X, y

    elif isinstance(params[0], tuple):
        combinations = list(product(*[values for _, values in params]))

        all_X = []
        all_y = []

        for c in combinations:
            for i, v in enumerate(c):
                params[i][0].set_value(v)

            X, y = make_parameterized_classification(
                p0, p1,
                n_samples // len(combinations),
                [p for p, _ in params])

            all_X.append(X)
            all_y.append(y)

        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        return X, y

    else:
        raise ValueError
