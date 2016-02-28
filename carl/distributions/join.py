# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import theano
import theano.tensor as T

from sklearn.utils import check_random_state

from . import DistributionMixin
from . import TheanoDistribution
from .base import check_parameter


class Join(TheanoDistribution):
    def __init__(self, components, random_state=None, optimizer=None):
        super(Join, self).__init__(random_state=random_state,
                                   optimizer=optimizer)

        self.components = components

        for i, component in enumerate(components):
            # Add component parameters, constants and observeds
            if isinstance(component, TheanoDistribution):
                for p_i in component.parameters_:
                    self.parameters_.add(p_i)
                for c_i in component.constants_:
                    self.constants_.add(c_i)
                for o_i in component.observeds_:
                    self.observeds_.add(o_i)

        # Derive and overide pdf, nnlf and cdf analytically if possible
        # XXX todo

    def rvs(self, n_samples, **kwargs):
        out = np.zeros((n_samples, self.ndim))
        start = 0

        for i, component in enumerate(self.components):
            out[:, start:start+component.ndim] = component.rvs(n_samples,
                                                               **kwargs)
            start += component.ndim

        return out

    @property
    def ndim(self):
        return sum([c.ndim for c in self.components])
