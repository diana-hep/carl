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
from .base import bound


class Normal(DistributionMixin):
    def __init__(self, random_state=None, mu=0.0, sigma=1.0):
        super(Normal, self).__init__(mu=mu, sigma=sigma,
                                     random_state=random_state, optimizer=None)

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
        self.ppf_ = (self.mu + \
                     np.sqrt(2.) * self.sigma * T.erfinv(2. * self.p - 1.))
        self.make_(self.ppf_, "ppf", args=[self.p])
