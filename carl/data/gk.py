# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.utils import check_random_state

from ..distributions import TheanoDistribution


class GK(TheanoDistribution):
    def __init__(self, A, B, g, k, c=0.8):
        super(GK, self).__init__(A=A, B=B, g=g, k=k, c=c)

    def rvs(self, n_samples, random_state=None, **kwargs):
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
