# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import theano
import theano.tensor as T

from sklearn.utils import check_random_state

from . import TheanoDistribution
from .base import check_parameter


class LinearTransform(TheanoDistribution):
    def __init__(self, p, A, random_state=None, optimizer=None):
        super(LinearTransform, self).__init__(random_state=random_state,
                                              optimizer=optimizer)

        self.p = p
        self.A = A
        self.inv_A = np.linalg.inv(A)

        if isinstance(p, TheanoDistribution):
            for p_i in p.parameters_:
                self.parameters_.add(p_i)
            for c_i in p.constants_:
                self.constants_.add(c_i)
            for o_i in p.observeds_:
                self.observeds_.add(o_i)

        # Derive and overide pdf, nnlf and cdf analytically if possible
        # XXX todo

    def pdf(self, X, **kwargs):
        return self.p.pdf(np.dot(self.inv_A, X.T).T, **kwargs)

    def nnlf(self, X, **kwargs):
        return self.p.nnlf(np.dot(self.inv_A, X.T).T, **kwargs)

    def rvs(self, n_samples, **kwargs):
        out = self.p.rvs(n_samples, **kwargs)
        return np.dot(self.A, out.T).T

    @property
    def ndim(self):
        return self.p.ndim
