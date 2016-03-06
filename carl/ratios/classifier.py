# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import copy
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.utils import check_random_state

from ..learning import as_classifier
from .base import DensityRatioMixin


class ClassifierRatio(BaseEstimator, DensityRatioMixin):
    def __init__(self, base_estimator, random_state=None):
        self.base_estimator = base_estimator
        self.random_state = random_state

    def fit(self, X=None, y=None, numerator=None, denominator=None,
            n_samples=None, **kwargs):
        # Check for identity
        self.identity_ = (numerator is not None) and (numerator is denominator)

        if self.identity_:
            return self

        # Build training data
        rng = check_random_state(self.random_state)

        if (numerator is not None and denominator is not None and
                n_samples is not None):
            X = np.vstack(
                (numerator.rvs(n_samples // 2,
                               random_state=rng, **kwargs),
                 denominator.rvs(n_samples - (n_samples // 2),
                                 random_state=rng, **kwargs)))
            y = np.zeros(n_samples, dtype=np.int)
            y[n_samples // 2:] = 1

        elif X is not None and y is not None:
            pass  # Use given X and y

        else:
            raise ValueError

        # Fit base estimator
        clf = clone(self.base_estimator)

        if isinstance(clf, RegressorMixin):
            clf = as_classifier(clf)

        self.classifier_ = clf.fit(X, y)

        return self

    def predict(self, X, log=False, **kwargs):
        if self.identity_:
            if log:
                return np.zeros(len(X))
            else:
                return np.ones(len(X))

        else:
            p = self.classifier_.predict_proba(X)

            if log:
                return np.log(p[:, 0]) - np.log(p[:, 1])
            else:
                return np.divide(p[:, 0], p[:, 1])
