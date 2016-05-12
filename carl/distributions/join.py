# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import theano

from sklearn.utils import check_random_state

from . import TheanoDistribution


class Join(TheanoDistribution):
    """Joint distribution.

    This class can be used to define a joint distribution
    `p(x, y, z, ...) = p_0(x) * p_1(y) * p_2(z) * ...`, where `p_i` are
    themselves distributions.
    """

    def __init__(self, components):
        """Constructor.

        Parameters
        ----------
        * `components` [list of `DistributionMixin`]:
            The components to join together.
        """
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

        # Derive and overide pdf and nll analytically if possible
        if all([hasattr(c, "pdf_") for c in self.components]):
            # pdf
            c0 = self.components[0]
            self.pdf_ = theano.clone(c0.pdf_, {c0.X: self.X[:, 0:c0.ndim]})
            start = c0.ndim

            for c in self.components[1:]:
                self.pdf_ *= theano.clone(
                    c.pdf_, {c.X: self.X[:, start:start+c.ndim]})
                start += c.ndim

            self._make(self.pdf_, "pdf")

        if all([hasattr(c, "nll_") for c in self.components]):
            # nll
            c0 = self.components[0]
            self.nll_ = theano.clone(c0.nll_, {c0.X: self.X[:, 0:c0.ndim]})
            start = c0.ndim

            for c in self.components[1:]:
                self.nll_ += theano.clone(
                    c.nll_, {c.X: self.X[:, start:start+c.ndim]})
                start += c.ndim

            self._make(self.nll_, "nll")

    def pdf(self, X, **kwargs):
        out = np.ones(len(X))
        start = 0

        for i, component in enumerate(self.components):
            out *= component.pdf(X[:, start:start+component.ndim], **kwargs)
            start += component.ndim

        return out

    def nll(self, X, **kwargs):
        out = np.zeros(len(X))
        start = 0

        for i, component in enumerate(self.components):
            out += component.nll(X[:, start:start+component.ndim], **kwargs)
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
        if hasattr(self, "nll_"):
            return super(Join, self).fit(X, **kwargs)
        else:
            raise NotImplementedError

    def cdf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    def ppf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    @property
    def ndim(self):
        return sum([c.ndim for c in self.components])
