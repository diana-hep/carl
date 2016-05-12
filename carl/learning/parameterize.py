"""This module implements tools for parameterized supervised learning."""

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
from sklearn.utils import check_random_state


class ParameterStacker(BaseEstimator, TransformerMixin):
    """Stack current parameter values as additional features."""

    def __init__(self, params):
        """Constructor.

        Parameters
        ----------
        * `params` [list of Theano shared variables]:
            The parameters.
        """
        self.params = params

    def transform(self, X, y=None):
        """Stack current parameter values as additional features.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `Xt` [array, shape=(n_samples, n_features+len(params))]:
            The horizontal concatenation of X with the current parameter
            values, added as new columns.
        """
        Xp = np.empty((len(X), len(self.params)))

        for i, p in enumerate(self.params):
            Xp[:, i] = p.eval()

        return np.hstack((X, Xp))


class _ParameterizedEstimator(BaseEstimator):
    """Parameterize a Scikit-Learn estimator.

    This wrapper can be used to learn a parameterized classification or
    regression problem, where parameter values are automatically added
    as additional features.
    """

    def __init__(self, base_estimator, params):
        """Constructor.

        Parameters
        ----------
        * `base_estimator` [`BaseEstimator`]:
            The estimator to parameterize.

        * `params` [list of Theano shared variables]:
            The parameters.
        """
        self.base_estimator = base_estimator
        self.params = params

    def _validate_X(self, X):
        if X.shape[1] == self.n_features_:
            X = self.stacker_.transform(X)
        assert X.shape[1] == self.n_features_ + len(self.params)

        return X

    def fit(self, X, y):
        """Fit estimator on parameterized data.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features+len(params))]:
            The samples, concatenated with the corresponding parameter values.

        * `y` [array-like, shape=(n_samples,)]:
            The output values.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
        self.stacker_ = ParameterStacker(self.params)

        # XXX: this assumes that X is extended with parameters
        self.n_features_ = X.shape[1] - len(self.params)
        self.estimator_ = clone(self.base_estimator).fit(X, y)

        return self

    def predict(self, X):
        """Predict the targets for `X`.

        Parameter values are automatically appended from the current state
        of the parameters if those are not provided with `X`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features) or
                           shape=(n_samples, n_features+len(params))]:
            The samples.

        Returns
        -------
        * `y` [array, shape=(n_samples,)]:
            The predicted output values.
        """
        return self.estimator_.predict(self._validate_X(X))


class ParameterizedClassifier(_ParameterizedEstimator, ClassifierMixin):
    """Parameterize a Scikit-Learn classifier.

    This wrapper can be used to learn a parameterized classification problem,
    where parameter values are automatically added as additional features.
    """

    def predict_proba(self, X):
        """Predict the posterior probabilities of classification for X.

        Parameter values are automatically appended from the current state
        of the parameters if those are not provided with X.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features) or
                           shape=(n_samples, n_features+len(params))]:
            The samples.

        Returns
        -------
        * `probas` [array, shape=(n_samples, n_classes)]:
            The predicted probabilities.
        """
        return self.estimator_.predict_proba(self._validate_X(X))


class ParameterizedRegressor(_ParameterizedEstimator, RegressorMixin):
    """Parameterize a Scikit-Learn regressor.

    This wrapper can be used to learn a parameterized regression problem,
    where parameter values are automatically added as additional features.
    """

    pass


def make_parameterized_classification(p0, p1, n_samples, params,
                                      random_state=None):
    """Generate parameterized classification data.

    This function generates parameterized classification data, by enumerating
    all possible combinations of provided parameter values and producing
    samples in equal number from `p0` and `p1`.

    Parameters
    ----------
    * `p0` [`DistributionMixin`]:
        The distribution to draw samples from class 0.

    * `p1` [`DistributionMixin`]:
        The distribution to draw samples from class 1.

    * `n_samples` [integer]:
        The total number of samples to generate.

    * `params` [list of pairs (theano shared variables, list of values) or
                list of theano shared variables]:
        The list of parameters and the corresponding values to generate
        samples for. If only a list of theano shared variables is given, then
        generate samples using the current parameter values.

    * `random_state` [integer or RandomState object]:
        The random seed.

    Returns
    -------
    * `X` [array, shape=(n_samples, n_features+len(params))]:
        The generated training data, as sample features and concatenated
        parameter values.

    * `y` [array, shape=(n_samples,)]:
        The labels.
    """
    rng = check_random_state(random_state)

    if not isinstance(params[0], tuple):
        X0 = p0.rvs(n_samples // 2, random_state=rng)
        X1 = p1.rvs(n_samples - (n_samples // 2), random_state=rng)
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
                [p for p, _ in params],
                random_state=rng)

            all_X.append(X)
            all_y.append(y)

        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        return X, y

    else:
        raise ValueError
