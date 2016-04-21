# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.utils import check_random_state

from . import TheanoDistribution


class LinearTransform(TheanoDistribution):
    """Apply a linear transformation `u = A*x` to `x ~ p`."""

    def __init__(self, p, A):
        """Constructor.

        Parameters
        ----------
        * `p` [`DistributionMixin`]:
            The base distribution.

        * `A` [array, shape=(p.ndim, p.ndim)]:
            The linear operator.
        """
        super(LinearTransform, self).__init__()

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

        # Derive and overide pdf, nll and cdf analytically if possible
        # XXX todo

    def pdf(self, X, **kwargs):
        return self.p.pdf(np.dot(self.inv_A, X.T).T, **kwargs)

    def nll(self, X, **kwargs):
        return self.p.nll(np.dot(self.inv_A, X.T).T, **kwargs)

    def ppf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    def cdf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    def rvs(self, n_samples, random_state=None, **kwargs):
        rng = check_random_state(random_state)
        out = self.p.rvs(n_samples, random_state=rng, **kwargs)
        return np.dot(self.A, out.T).T

    @property
    def ndim(self):
        return self.p.ndim
