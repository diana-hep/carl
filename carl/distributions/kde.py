# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

from scipy.stats import gaussian_kde
from sklearn.utils import check_array

from .base import DistributionMixin

# XXX: replace with a wrapper of sklearn.neighbors.KernelDensity instead?


class KernelDensity(DistributionMixin):
    """Kernel density estimation.

    This distribution supports 1D data only.
    """

    def __init__(self, bandwidth=None):
        """Constructor.

        Parameters
        ----------
        * `bandwidth` [string or float, optional]:
            The method used to calculate the estimator bandwidth.
        """
        self.bandwidth = bandwidth

    def pdf(self, X, **kwargs):
        X = check_array(X)
        return self.kde_.pdf(X.T)

    def nll(self, X, **kwargs):
        X = check_array(X)
        return -self.kde_.logpdf(X.T)

    def rvs(self, n_samples, random_state=None, **kwargs):
        # XXX gaussian_kde uses Numpy global random state...
        return self.kde_.resample(n_samples).T

    def fit(self, X, **kwargs):
        """Fit the KDE estimator to data.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
        X = check_array(X).T
        self.kde_ = gaussian_kde(X, bw_method=self.bandwidth)
        return self

    def cdf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError

    def ppf(self, X, **kwargs):
        """Not supported."""
        raise NotImplementedError
