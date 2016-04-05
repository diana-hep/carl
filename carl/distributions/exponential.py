# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import theano.tensor as T

from . import TheanoDistribution
from .base import bound


class Exponential(TheanoDistribution):
    def __init__(self, inverse_scale=1.0):
        super(Exponential, self).__init__(inverse_scale=inverse_scale)

        # pdf
        self.pdf_ = T.switch(
            T.lt(self.X, 0.),
            0.,
            self.inverse_scale * T.exp(-self.inverse_scale * self.X)).ravel()
        self.make_(self.pdf_, "pdf")

        # -log pdf
        self.nll_ = bound(
            -T.log(self.inverse_scale) + self.inverse_scale * self.X,
            np.inf,
            self.inverse_scale > 0.).ravel()
        self.make_(self.nll_, "nll")

        # cdf
        self.cdf_ = (1. - T.exp(-self.inverse_scale * self.X)).ravel()
        self.make_(self.cdf_, "cdf")

        # ppf
        self.ppf_ = -T.log(1. - self.p) / self.inverse_scale
        self.make_(self.ppf_, "ppf", args=[self.p])
