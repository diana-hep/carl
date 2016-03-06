# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.linalg as linalg

from sklearn.utils import check_random_state

from . import TheanoDistribution
from .base import bound


class Normal(TheanoDistribution):
    def __init__(self, mu=0.0, sigma=1.0):
        super(Normal, self).__init__(mu=mu, sigma=sigma)

        # pdf
        self.pdf_ = (
            (1. / np.sqrt(2. * np.pi)) / self.sigma *
            T.exp(-(self.X - self.mu) ** 2 / (2. * self.sigma ** 2))).ravel()
        self.make_(self.pdf_, "pdf")

        # -log pdf
        self.nnlf_ = bound(
            T.log(self.sigma) + T.log(np.sqrt(2. * np.pi)) +
            (self.X - self.mu) ** 2 / (2. * self.sigma ** 2),
            np.inf,
            self.sigma > 0.).ravel()
        self.make_(self.nnlf_, "nnlf")

        # cdf
        self.cdf_ = 0.5 * (1. + T.erf((self.X - self.mu) /
                                      (self.sigma * np.sqrt(2.)))).ravel()
        self.make_(self.cdf_, "cdf")

        # ppf
        self.ppf_ = (self.mu +
                     np.sqrt(2.) * self.sigma * T.erfinv(2. * self.p - 1.))
        self.make_(self.ppf_, "ppf", args=[self.p])


class MultivariateNormal(TheanoDistribution):
    def __init__(self, mu, sigma, random_state=None):
        super(MultivariateNormal, self).__init__(mu=mu, sigma=sigma)
        # XXX: The SDP-ness of sigma should be check upon changes

        # ndim
        self.ndim_ = self.mu.shape[0]
        self.make_(self.ndim_, "ndim_func_", args=[])

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
        self.make_(self.pdf_, "pdf")

        # -log pdf
        self.nnlf_ = -T.log(self.pdf_)  # XXX: for sure this can be better
        self.make_(self.nnlf_, "nnlf")

        # self.rvs_
        self.make_(T.dot(L, self.X.T).T + self.mu, "rvs_func_")

    def rvs(self, n_samples, random_state=None, **kwargs):
        rng = check_random_state(random_state)
        X = rng.randn(n_samples, self.ndim)
        return self.rvs_func_(X, **kwargs)

    @property
    def ndim(self):
        return self.ndim_func_()[None][0]
