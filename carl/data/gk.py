# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.utils import check_random_state

from ..distributions import TheanoDistribution


class GK(TheanoDistribution):
    """g-and-k distribution generator.

    The g-and-k distribution is defined as `x = A + B *
    (1 + c * (1 - exp(-g * z)) / (1 + exp(-g * z))) * ((1 + z ** 2) ** k) * z`,
    where `z ~ N(0, 1)`.

    Reference
    ---------
    * Haynes, Michele A., H. L. MacGillivray, and K. L. Mengersen.
      "Robustness of ranking and selection rules using generalised g-and-k
      distributions." Journal of Statistical Planning and Inference 65.1
      (1997): 45-66.
    """

    def __init__(self, A, B, g, k, c=0.8):
        """Constructor.

        Parameters
        ----------
        * `A` [float or theano shared variable]
        * `B` [float or theano shared variable]
        * `g` [float or theano shared variable]
        * `k` [float or theano shared variable]
        * `c` [float or theano shared variable, default=0.8]
        """
        super(GK, self).__init__(A=A, B=B, g=g, k=k, c=c)

    def rvs(self, n_samples, random_state=None, **kwargs):
        """Generate random samples.

        Parameters
        ----------
        * `n_samples` [int]:
            The number of samples to generate.

        * `random_state` [integer or RandomState object]:
            The random seed.
        """
        rng = check_random_state(random_state)

        A = self.A.eval()
        B = self.B.eval()
        g = self.g.eval()
        k = self.k.eval()
        c = self.c.eval()

        z = rng.randn(n_samples).reshape(-1, 1)

        Q = A + B * (1 + c * (1 - np.exp(-g * z)) /
                             (1 + np.exp(-g * z))) * ((1 + z ** 2) ** k) * z

        return Q
