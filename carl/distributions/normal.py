# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import theano.tensor as T
import theano.sandbox.linalg as linalg

from sklearn.utils import check_random_state

from . import TheanoDistribution
from .base import bound


class Normal(TheanoDistribution):
    """Normal distribution.

    This distribution supports 1D data only.
    """

    def __init__(self, mu=0.0, sigma=1.0):
        """Constructor.

        Parameters
        ----------
        * `mu` [float]:
            The distribution mean.

        * `sigma` [float]:
            The distribution standard deviation.
        """
        super(Normal, self).__init__(mu=mu, sigma=sigma)

        # pdf
        self.pdf_ = (
            (1. / np.sqrt(2. * np.pi)) / self.sigma *
            T.exp(-(self.X - self.mu) ** 2 / (2. * self.sigma ** 2))).ravel()
        self._make(self.pdf_, "pdf")

        # -log pdf
        self.nll_ = bound(
            T.log(self.sigma) + T.log(np.sqrt(2. * np.pi)) +
            (self.X - self.mu) ** 2 / (2. * self.sigma ** 2),
            np.inf,
            self.sigma > 0.).ravel()
        self._make(self.nll_, "nll")

        # cdf
        self.cdf_ = 0.5 * (1. + T.erf((self.X - self.mu) /
                                      (self.sigma * np.sqrt(2.)))).ravel()
        self._make(self.cdf_, "cdf")

        # ppf
        self.ppf_ = (self.mu +
                     np.sqrt(2.) * self.sigma * T.erfinv(2. * self.p - 1.))
        self._make(self.ppf_, "ppf", args=[self.p])


class MultivariateNormal(TheanoDistribution):
    """Multivariate normal distribution."""

    def __init__(self, mu, sigma):
        """Constructor.

        Parameters
        ----------
        * `mu` [1d array]:
            The means.

        * `sigma` [2d array]:
            The covariance matrix.
        """
        super(MultivariateNormal, self).__init__(mu=mu, sigma=sigma)
        # XXX: The SDP-ness of sigma should be check upon changes

        # ndim
        self.ndim_ = self.mu.shape[0]
        self._make(self.ndim_, "ndim_func_", args=[])

        # pdf
        L = linalg.cholesky(self.sigma)
        sigma_det = linalg.det(self.sigma)  # XXX: compute from L instead
        sigma_inv = linalg.matrix_inverse(self.sigma)  # XXX: idem

        self.pdf_ = (
            (1. / T.sqrt((2. * np.pi) ** self.ndim_ * T.abs_(sigma_det))) *
            T.exp(-0.5 * T.sum(T.mul(T.dot(self.X - self.mu,
                                           sigma_inv),
                                     self.X - self.mu),
                               axis=1))).ravel()
        self._make(self.pdf_, "pdf")

        # -log pdf
        self.nll_ = -T.log(self.pdf_)  # XXX: for sure this can be better
        self._make(self.nll_, "nll")

        # self.rvs_
        self._make(T.dot(L, self.X.T).T + self.mu, "rvs_func_")

    def rvs(self, n_samples, random_state=None, **kwargs):
        rng = check_random_state(random_state)
        X = rng.randn(n_samples, self.ndim)
        return self.rvs_func_(X, **kwargs)

    def cdf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    def ppf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    @property
    def ndim(self):
        return self.ndim_func_()[None][0]
