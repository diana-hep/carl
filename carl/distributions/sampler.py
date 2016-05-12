# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.utils import check_random_state

from .base import DistributionMixin


class Sampler(DistributionMixin):
    """Sampler.

    This class can be used in the likelihood-free setup to provide an
    implementation of the `rvs` method on top of known data.
    """

    def fit(self, X, sample_weight=None, **kwargs):
        """Fit.

        Note that calling `fit` is necessary to store a reference to the
        data `X`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
        self.X_ = X
        self.ndim_ = X.shape[1]
        self.sample_weight_ = sample_weight

        return self

    def rvs(self, n_samples, random_state=None, **kwargs):
        rng = check_random_state(random_state)

        if self.sample_weight_ is None:
            w = np.ones(len(self.X_))
        else:
            w = self.sample_weight_

        w = w / w.sum()
        indices = np.searchsorted(np.cumsum(w), rng.rand(n_samples))

        return self.X_[indices]

    def pdf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    def nll(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    def ppf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    def cdf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    def score(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    @property
    def ndim(self):
        return self.ndim_
