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
from .base import bound


class Exponential(DistributionMixin):
    def __init__(self, random_state=None, inv_scale=1.0):
        super(Exponential, self).__init__(inv_scale=inv_scale,
                                          random_state=random_state,
                                          optimizer=None)

        # pdf
        self.pdf_ = self.inv_scale * T.exp(-self.inv_scale * self.X)
        self.make_(self.pdf_, "pdf")

        # -log pdf
        self.nnlf_ = bound(-T.log(self.inv_scale) + self.inv_scale * self.X,
                           np.inf, self.inv_scale > 0)
        self.make_(self.nnlf_, "nnlf")

        # cdf
        self.cdf_ = 1 - T.exp(-self.inv_scale * self.X)
        self.make_(self.cdf_, "cdf")

        # rvs
        n_samples = T.iscalar()
        rng = check_random_state(self.random_state)
        u = rng.uniform(size=(n_samples, 1), low=0., high=1.)
        rvs_ = -T.log(1 - u) / self.inv_scale
        func = theano.function([n_samples] +
                               [theano.Param(v, name=v.name)
                                for v in self.observeds_ if v is not self.X],
                               rvs_)

        def rvs(n_samples, **kwargs):
            return func(n_samples, **kwargs)

        setattr(self, "rvs", rvs)
