"""This module implements commons for density ratio estimation."""

# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import warnings

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.metrics import mean_squared_error


class DensityRatioMixin:
    """Density ratio mixin.

    This class defines the common API for density ratio estimators.
    """

    def fit(self, X=None, y=None, numerator=None,
            denominator=None, n_samples=None, **kwargs):
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
        raise NotImplementedError

    def nllr(self, X, **kwargs):
        """Negative log-likelihood ratio.

        This method is a shortcut for `-ratio.predict(X, log=True).sum()`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `nllr` [float]:
            The negative log-likelihood ratio.
        """
        ratios = self.predict(X, log=True, **kwargs)
        mask = np.isfinite(ratios)

        if mask.sum() < len(ratios):
            warnings.warn("r(X) contains non-finite values.")

        return -np.sum(ratios[mask])

    def score(self, X, y, finite_only=True, **kwargs):
        """Negative MSE between predicted and known ratios.

        The higher, the better.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        * `y` [array-like, shape=(n_samples,)]:
            The known ratios for `X`.

        * `finite_only` [boolean, default=True]:
            Evaluate the MSE only over finite ratios.

        Returns
        -------
        * `score` [float]:
            The negative MSE.
        """
        ratios = self.predict(X, **kwargs)

        if finite_only:
            mask = np.isfinite(ratios)
            y = y[mask]
            ratios = ratios[mask]

        return -mean_squared_error(y, ratios)


class KnownDensityRatio(DensityRatioMixin, BaseEstimator):
    """Density ratio for known densities `p0` and `p1`.

    This class cannot be used in the likelihood-free setup. It requires
    numerator and denominator distributions to implement the `pdf` and `nll`
    methods.
    """

    def __init__(self, numerator, denominator):
        """Constructor.

        Parameters
        ----------
        * `numerator` [`DistributionMixin`]:
            The numerator distribution.
            This object is required to implement the `pdf` and `nll` methods.

        * `denominator` [`DistributionMixin`]:
            The denominator distribution.
            This object is required to implement the `pdf` and `nll` methods.
        """
        self.numerator = numerator
        self.denominator = denominator

    def fit(self, X=None, y=None, numerator=None,
            denominator=None, n_samples=None, **kwargs):
        """Do nothing.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
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
        if not log:
            return self.numerator.pdf(X) / self.denominator.pdf(X)

        else:
            return -self.numerator.nll(X) + self.denominator.nll(X)


class InverseRatio(DensityRatioMixin, BaseEstimator):
    """Inverse a density ratio.

    This class can be used to model the inverse `1 / r(x) = p1(x) / p0(x)`
    of a given base ratio `r(x) = p0(x) / p1(x)`.
    """

    def __init__(self, base_ratio):
        """Constructor.

        Parameters
        ----------
        * `base_ratio` [`DensityRatioMixin`]:
            The base ratio to inverse.
        """
        self.base_ratio = base_ratio

    def fit(self, X=None, y=None, numerator=None,
            denominator=None, n_samples=None, **kwargs):
        """Fit the inversed density ratio estimator.

        The inversed density ratio estimator `1 / r(x) = p1(x) / p0(x)` can be
        fit either

        - from data, using `fit(X, y)` or
        - from distributions, using
          `fit(numerator=p1, denominator=p0, n_samples=N)`

        Note that this object does not need to be fit if `base_ratio`
        is already fit.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features), optional]:
            Training data.

        * `y` [array-like, shape=(n_samples,), optional]:
            Labels. Samples labeled with `y=1` correspond to data from the
            numerator distribution, while samples labeled with `y=0` correspond
            data from the denominator distribution.

        * `numerator` [`DistributionMixin`, optional]:
            The numerator distribution `p1`, if `X` and `y` are not provided.

        * `denominator` [`DistributionMixin`, optional]:
            The denominator distribution `p0`, if `X` and `y` are not provided.

        * `n_samples` [integer, optional]
            The total number of samples to draw from the numerator and
            denominator distributions, if `X` and `y` are not provided.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
        if X is not None and y is not None:
            self.base_ratio.fit(X=X, y=1 - y)
        else:
            self.base_ratio.fit(numerator=numerator, denominator=denominator,
                                n_samples=n_samples)

        return self

    def predict(self, X, log=False, **kwargs):
        """Predict the inverse density ratio `1 / r(x_i)` for all `x_i` in `X`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        * `log` [boolean, default=False]:
            If true, return the negative log-ratio
            `-log r(x) = log(p1(x)) - log(p0(x))`.

        Returns
        -------
        * `r` [array, shape=(n_samples,)]:
            The predicted inverse ratio `1 / r(X)`.
        """
        if log:
            return -self.base_ratio.predict(X, log=True, **kwargs)
        else:
            return 1. / self.base_ratio.predict(X, log=False, **kwargs)


class DecomposedRatio(DensityRatioMixin, BaseEstimator):
    """Decompose a ratio of mixtures into a weighted sum of sub-ratios.

    If numerator `p0` and denominator `p1` distributions are known to be
    mixtures `p0(x) = \sum_i w_i p0_i(x)` and `p1(x) = \sum_j w_j p1_j(x)`,
    then this  class can be used to decompose the ratio `r(x) = p0(x) / p1(x)`
    into the equivalent form `r(x) = \sum_i [\sum_j r_ji(x)]^-1` where
    `r_ji(x)` are sub-ratios for `p1_j(x) / p0_i(x)`. This usually allows
    for more accurate estimates.

    The given numerator and denominator distributions are expected to
    follow the `Mixture` API.
    """

    def __init__(self, base_ratio):
        """Constructor.

        Parameters
        ----------
        * `base_ratio` [`DensityRatioMixin`]:
            The base ratio to use for modeling the sub-ratios `r_ji(x)`.
        """
        self.base_ratio = base_ratio

    def fit(self, X=None, y=None, numerator=None,
            denominator=None, n_samples=None, **kwargs):
        """Fit the decomposed density ratio estimator.

        The decomposed density ratio estimator can be fit only from
        distributions, using `fit(numerator=p0, denominator=p1, n_samples=N)`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features), optional]:
            Not supported.

        * `y` [array-like, shape=(n_samples,), optional]:
            Not supported

        * `numerator` [`Mixture`, optional]:
            The numerator mixture `p0`.

        * `denominator` [`Mixture`, optional]:
            The denominator mixture `p1`.

        * `n_samples` [integer, optional]
            The total number of samples to draw from the numerator and
            denominator distributions.
            If numerator and denominator are made of `p` and `q` components,
            then `n_samples // (p * q)` samples are drawn per sub-ratio
            `r_ji(x)` to fit.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
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
