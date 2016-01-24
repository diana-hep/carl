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
from .base import bound


class Mixture(TheanoDistribution):
    def __init__(self, components, weights=None,
                 random_state=None, optimizer=None):
        super(Mixture, self).__init__(random_state=random_state,
                                      optimizer=optimizer)

        self.components = components
        self.weights = []

        # Check component and weights
        # XXX check that ndim fit

        if weights is None:
            weights = [1. / len(components)] * (len(components) - 1)

        if len(weights) == len(components) - 1:
            weights.append(None)

        if len(weights) != len(components):
            raise ValueError("Mixture components and weights must be in "
                             "equal number.")

        for i, (component, weight) in enumerate(zip(components, weights)):
            # Add component parameters, constants and observeds
            if isinstance(component, TheanoDistribution):
                for p_i in component.parameters_:
                    self.parameters_.add(p_i)
                for c_i in component.constants_:
                    self.constants_.add(c_i)
                for o_i in component.observeds_:
                    self.observeds_.add(o_i)

            # Validate weights
            if weight is not None:
                v, p, c, o = check_parameter("param_w{}".format(i), weight)
                self.weights.append(v)

                for p_i in p:
                    self.parameters_.add(p_i)
                for c_i in c:
                    self.constants_.add(c_i)
                for o_i in o:
                    self.observeds_.add(o_i)

            else:
                assert i == len(self.components) - 1
                w_last = 1.

                for w_i in self.weights:
                    w_last = w_last - w_i

                self.weights.append(w_last)

        # Normalize weights
        normalizer = self.weights[0]
        for w in self.weights[1:]:
            normalizer += w
        self.weights = [w / normalizer for w in self.weights]

        # Derive and overide pdf, nnlf and cdf analytically if possible
        if all([hasattr(c, "pdf_") for c in self.components]):
            # pdf
            self.pdf_ = self.weights[0] * self.components[0].pdf_
            for i in range(1, len(self.components)):
                self.pdf_ += self.weights[i] * self.components[i].pdf_
            self.make_(self.pdf_, "pdf")

            # -log pdf
            self.nnlf_ = self.weights[0] * self.components[0].pdf_
            for i in range(1, len(self.components)):
                self.nnlf_ += self.weights[i] * self.components[i].pdf_
            self.nnlf_ = -T.log(self.nnlf_)
            self.make_(self.nnlf_, "nnlf")

        if all([hasattr(c, "cdf_") for c in self.components]):
            # cdf
            self.cdf_ = self.weights[0] * self.components[0].cdf_
            for i in range(1, len(self.components)):
                self.cdf_ += self.weights[i] * self.components[i].cdf_
            self.make_(self.cdf_, "cdf")

        # Weight evaluation function
        self.make_(T.stack(*self.weights), "compute_weights", args=[])

    def pdf(self, X, **kwargs):
        weights = self.compute_weights(**kwargs)
        p = weights[0] * self.components[0].pdf(X, **kwargs)

        for i in range(1, len(self.components)):
            p += weights[i] * self.components[i].pdf(X, **kwargs)

        return p

    def nnlf(self, X, **kwargs):
        return -np.log(self.pdf(X, **kwargs))

    def cdf(self, X, **kwargs):
        weights = self.compute_weights(**kwargs)
        c = weights[0] * self.components[0].cdf(X, **kwargs)

        for i in range(1, len(self.components)):
            c += weights[i] * self.components[i].cdf(X, **kwargs)

        return c

    def rvs(self, n_samples, **kwargs):
        rng = check_random_state(self.random_state)
        indices = rng.multinomial(1,
                                  pvals=self.compute_weights(**kwargs),
                                  size=n_samples)
        out = np.zeros((n_samples, self.ndim()))

        for j in range(len(self.components)):
            mask = np.where(indices[:, j])[0]
            if len(mask) > 0:
                out[mask, :] = self.components[j].rvs(n_samples=len(mask),
                                                      **kwargs)

        return out

    def fit(self, X, y=None, **kwargs):
        if all([hasattr(c, "nnlf_") for c in self.components]):
            return super(Mixture, self).fit(X, y=y, **kwargs)
        else:
            raise NotImplementedError

    def ndim(self, **kwargs):
        return self.components[0].ndim()
