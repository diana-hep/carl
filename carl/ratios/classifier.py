"""This module implements classifier-based density ratio estimation."""

# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.utils import check_random_state

from .base import DensityRatioMixin
from ..learning import as_classifier


class ClassifierRatio(BaseEstimator, DensityRatioMixin):
    """Classifier-based density ratio estimator.

    This class approximates a density ratio `r(x) = p0(x) / p1(x)` as
    `s(x) / 1 - s(x)`, where `s` is a classifier trained to distinguish
    samples `x ~ p0` from samples `x ~ p1`, and where `s(x)` is the
    approximated probability of `p0(x) / (p0(x) + p1(x))`.
    """

    def __init__(self, base_estimator, random_state=None):
        """Constructor.

        Parameters
        ----------
        * `base_estimator` [`BaseEstimator`]:
            A scikit-learn classifier or regressor.

        * `random_state` [integer or RandomState object]:
            The random seed.
        """
        self.base_estimator = base_estimator
        self.random_state = random_state

    def fit(self, X=None, y=None, numerator=None, denominator=None,
            n_samples=None, **kwargs):
        """Fit the density ratio estimator.

        The density ratio estimator `r(x) = p0(x) / p1(x)` can be fit either

        - from data, using `fit(X, y)` or
        - from distributions, using
          `fit(numerator=p0, denominator=p1, n_samples=N)`

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features), optional]:
            Training data.

        * `y` [array-like, shape=(n_samples,), optional]:
            Labels. Samples labeled with `y=0` correspond to data from the
            numerator distribution, while samples labeled with `y=1` correspond
            data from the denominator distribution.

        * `numerator` [`DistributionMixin`, optional]:
            The numerator distribution `p0`, if `X` and `y` are not provided.

        * `denominator` [`DistributionMixin`, optional]:
            The denominator distribution `p1`, if `X` and `y` are not provided.

        * `n_samples` [integer, optional]
            The total number of samples to draw from the numerator and
            denominator distributions, if `X` and `y` are not provided.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
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
        """Predict the density ratio `r(x_i)` for all `x_i` in `X`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        * `log` [boolean, default=False]:
            If true, return the log-ratio `log r(x) = log(p0(x)) - log(p1(x))`.

        Returns
        -------
        * `r` [array, shape=(n_samples,)]:
            The predicted ratio `r(X)`.
        """
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
