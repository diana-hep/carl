# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from carl.distributions import Normal
from carl.distributions import KernelDensity


def test_kde():
    # Test API
    p = Normal()
    X = p.rvs(10000, random_state=1)
    k = KernelDensity()
    k.fit(X)

    reals = np.linspace(-3, 3).reshape(-1, 1)
    assert np.mean(np.abs(p.pdf(reals) - k.pdf(reals))) < 0.05
    assert np.mean(np.abs(p.nll(reals) - k.nll(reals))) < 0.05

    # Test sampling
    X = k.rvs(10000, random_state=1)
    assert np.abs(np.mean(X)) < 0.05
