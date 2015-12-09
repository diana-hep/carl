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
from .base import check_parameter


class Mixture(DistributionMixin):
    def __init__(self, components, random_state=None, weights=None):
        super(Mixture, self).__init__(random_state=random_state)

        self.components = components
        self.weights = []

        if weights is None:
            weights = [1. / len(components)] * (len(components) - 1)

        if len(weights) == len(components) - 1:
            weights.append(None)

        for i, (component, weight) in enumerate(zip(components, weights)):
            for p_i in component.parameters_:
                self.parameters_.add(p_i)
            for c_i in component.constants_:
                self.constants_.add(c_i)
            for o_i in component.observeds_:
                self.observeds_.add(o_i)

            if weight is not None:  # XXX better placeholder for last/missing weight?
                v, p, c, o = check_parameter("w_{}".format(i), weight)
                self.weights.append(v)

                for p_i in p:
                    self.parameters_.add(p_i)
                for c_i in c:
                    self.constants._add(c_i)
                for o_i in c:
                    self.observeds_.add(o_i)

            else:
                w_last = 1.

                for w_i in self.weights:
                    w_last = w_last - w_i

                self.weights.append(w_last)

        # pdf
        self.pdf_ = self.weights[0] * self.components[0].pdf_
        for i in range(1, len(self.components)):
            self.pdf_ = self.pdf_ + self.weights[i] * self.components[i].pdf_
        self.make_(self.pdf_, "pdf")

        # cdf
        self.cdf_ = self.weights[0] * self.components[0].cdf_
        for i in range(1, len(self.components)):
            self.cdf_ = self.cdf_ + self.weights[i] * self.components[i].cdf_
        self.make_(self.cdf_, "cdf")
