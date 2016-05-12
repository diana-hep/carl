# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import theano.tensor as T

from sklearn.utils import check_random_state

from . import TheanoDistribution
from .base import check_parameter


class Mixture(TheanoDistribution):
    """Mix components into a mixture distribution.

    This class can be used to model a mixture distribution
    `p(x) = \sum_i w_i p_i(x)`, where `p_i` are themselves
    distributions and where `w_i` are the component weights.
    """

    def __init__(self, components, weights=None):
        """Constructor.

        Parameters
        ----------
        * `components` [list of `DistributionMixin`]:
            The components to mix together.

        * `weights` [list of floats or list of theano expressions]:
            The component weights.
        """

        super(Mixture, self).__init__()
        self.components = components
        self.weights = []

        # Check component and weights
        ndim = self.components[0].ndim

        for component in self.components[1:]:
            if ndim != component.ndim:
                raise ValueError("Mixture components must have the same "
                                 "number of dimensions.")

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

        # Derive and overide pdf, nll and cdf analytically if possible
        if all([hasattr(c, "pdf_") for c in self.components]):
            # pdf
            self.pdf_ = self.weights[0] * self.components[0].pdf_
            for i in range(1, len(self.components)):
                self.pdf_ += self.weights[i] * self.components[i].pdf_
            self._make(self.pdf_, "pdf")

            # -log pdf
            self.nll_ = self.weights[0] * self.components[0].pdf_
            for i in range(1, len(self.components)):
                self.nll_ += self.weights[i] * self.components[i].pdf_
            self.nll_ = -T.log(self.nll_)
            self._make(self.nll_, "nll")

        if all([hasattr(c, "cdf_") for c in self.components]):
            # cdf
            self.cdf_ = self.weights[0] * self.components[0].cdf_
            for i in range(1, len(self.components)):
                self.cdf_ += self.weights[i] * self.components[i].cdf_
            self._make(self.cdf_, "cdf")

        # Weight evaluation function
        self._make(T.stack(self.weights, axis=0), "compute_weights", args=[])

    def pdf(self, X, **kwargs):
        weights = self.compute_weights(**kwargs)
        p = weights[0] * self.components[0].pdf(X, **kwargs)

        for i in range(1, len(self.components)):
            p += weights[i] * self.components[i].pdf(X, **kwargs)

        return p

    def nll(self, X, **kwargs):
        return -np.log(self.pdf(X, **kwargs))

    def cdf(self, X, **kwargs):
        weights = self.compute_weights(**kwargs)
        c = weights[0] * self.components[0].cdf(X, **kwargs)

        for i in range(1, len(self.components)):
            c += weights[i] * self.components[i].cdf(X, **kwargs)

        return c

    def ppf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    def rvs(self, n_samples, random_state=None, **kwargs):
        rng = check_random_state(random_state)
        indices = rng.multinomial(1,
                                  pvals=self.compute_weights(**kwargs),
                                  size=n_samples)
        out = np.zeros((n_samples, self.ndim))

        for j in range(len(self.components)):
            mask = np.where(indices[:, j])[0]
            if len(mask) > 0:
                out[mask, :] = self.components[j].rvs(n_samples=len(mask),
                                                      random_state=rng,
                                                      **kwargs)

        return out

    def fit(self, X, **kwargs):
        if hasattr(self, "nll_"):
            return super(Mixture, self).fit(X, **kwargs)
        else:
            raise NotImplementedError

    @property
    def ndim(self):
        return self.components[0].ndim
