# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.utils import check_random_state

from ..distributions import TheanoDistribution


class Ricker(TheanoDistribution):
    """Ricker model.

    `N_t+1 = r * N_t * exp(-N_t + e_t); Y_t+1 ~ Poisson(phi * N_t+1)`,
    where `e_t` are iid normal deviates with zero mean and variance `sigma^2`.

    Reference
    ---------
    * Ricker, William E. "Stock and recruitment." Journal of the Fisheries
      Board of Canada 11.5 (1954): 559-623..
    """

    def __init__(self, log_r=3.8, sigma=0.3, phi=10.0):
        """Constructor.

        Parameters
        ----------
        * `log_r` [float or theano shared variable, default=3.8]
        * `sigma` [float or theano shared variable, default=0.3]
        * `phi` [float or theano shared variable, phi=10.0]
        """
        super(Ricker, self).__init__(log_r=log_r, sigma=sigma, phi=phi)

    def rvs(self, n_samples, random_state=None, **kwargs):
        """Generate random samples `Y_t`, for `t = 1:n_samples`.

        Parameters
        ----------
        * `n_samples` [int]:
            The number of samples to generate.

        * `random_state` [integer or RandomState object]:
            The random seed.
        """
        rng = check_random_state(random_state)

        log_r = self.log_r.eval()
        sigma = self.sigma.eval()
        phi = self.phi.eval()

        N_t = 1
        Y = []

        for t in range(1, n_samples + 1):
            e_t = sigma * rng.randn()
            N_t = np.exp(log_r) * N_t * np.exp(-N_t + e_t)
            Y_t = rng.poisson(phi * N_t)
            Y.append(Y_t)

        Y = np.array(Y).reshape(-1, 1)

        return Y
