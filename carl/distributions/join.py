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


class Join(TheanoDistribution):
    def __init__(self, components):
        super(Join, self).__init__()
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

        # Derive and overide pdf and nnlf analytically if possible
        if all([hasattr(c, "pdf_") for c in self.components]):
            # pdf
            c0 = self.components[0]
            self.pdf_ = theano.clone(c0.pdf_, {c0.X: self.X[:, 0:c0.ndim]})
            start = c0.ndim

            for c in self.components[1:]:
                self.pdf_ *= theano.clone(
                    c.pdf_, {c.X: self.X[:, start:start+c.ndim]})
                start += c.ndim

            self.make_(self.pdf_, "pdf")

        if all([hasattr(c, "nnlf_") for c in self.components]):
            # nnlf
            c0 = self.components[0]
            self.nnlf_ = theano.clone(c0.nnlf_, {c0.X: self.X[:, 0:c0.ndim]})
            start = c0.ndim

            for c in self.components[1:]:
                self.nnlf_ += theano.clone(
                    c.nnlf_, {c.X: self.X[:, start:start+c.ndim]})
                start += c.ndim

            self.make_(self.nnlf_, "nnlf")

    def pdf(self, X, **kwargs):
        out = np.ones(len(X))
        start = 0

        for i, component in enumerate(self.components):
            out *= component.pdf(X[:, start:start+component.ndim], **kwargs)
            start += component.ndim

        return out

    def nnlf(self, X, **kwargs):
        out = np.zeros(len(X))
        start = 0

        for i, component in enumerate(self.components):
            out += component.nnlf(X[:, start:start+component.ndim], **kwargs)
            start += component.ndim

        return out

    def rvs(self, n_samples, random_state=None, **kwargs):
        rng = check_random_state(random_state)
        out = np.zeros((n_samples, self.ndim))
        start = 0

        for i, component in enumerate(self.components):
            out[:, start:start+component.ndim] = component.rvs(
                n_samples, random_state=rng, **kwargs)
            start += component.ndim

        return out

    def fit(self, X, **kwargs):
        if hasattr(self, "nnlf_"):
            return super(Join, self).fit(X, **kwargs)
        else:
            raise NotImplementedError

    @property
    def ndim(self):
        return sum([c.ndim for c in self.components])
