# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import theano
import theano.tensor as T

from theano.gof import graph

from . import DistributionMixin
from .base import check_random_state


class Normal(DistributionMixin):
    def __init__(self, random_state=None, mu=0.0, sigma=1.0):
        super(Normal, self).__init__(random_state=random_state,
                                     mu=mu, sigma=sigma)

        # pdf
        self.pdf_ = 1. / (self.sigma * np.sqrt(2. * np.pi)) * \
                    T.exp(-(self.X - self.mu) ** 2 / (2. * self.sigma ** 2))
        self.make_(self.pdf_, "pdf")

        # -log pdf
        self.nnlf_ = T.log(self.sigma) + T.log(np.sqrt(2. * np.pi)) + \
                     (self.X - self.mu) ** 2 / (2. * self.sigma ** 2)
        self.make_(self.nnlf_, "nnlf")

        # cdf
        self.cdf_ = 0.5 * (1. + T.erf((self.X - self.mu) /
                                      (self.sigma * np.sqrt(2.))))
        self.make_(self.cdf_, "cdf")

        # rvs
        n_samples = T.iscalar()
        rng = check_random_state(self.random_state)
        u = rng.uniform(size=(n_samples, 1), low=0., high=1.)
        rvs_ = self.mu + self.sigma * np.sqrt(2.) * T.erfinv(2 * u - 1)
        func = theano.function([n_samples] +
                               [theano.Param(v, name=v.name)
                                for v in self.observeds_ if v is not self.X],
                               rvs_)

        def rvs(n_samples, **kwargs):
            return func(n_samples, **kwargs)

        setattr(self, "rvs", rvs)
