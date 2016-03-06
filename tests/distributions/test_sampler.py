# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from numpy.testing import assert_array_equal
from sklearn.utils import check_random_state

from carl.distributions import Sampler


def test_sampler():
    rng = check_random_state(1)
    X = rng.rand(10).reshape(-1, 1)

    sampler = Sampler()
    sampler.fit(X)
    X2 = sampler.rvs(1000, random_state=1)
    assert_array_equal(np.unique(X), np.unique(X2))
    assert sampler.ndim == 1

    sampler.fit(X, sample_weight=np.ones(len(X)) / len(X))
    X3 = sampler.rvs(1000, random_state=2)
    assert_array_equal(np.unique(X), np.unique(X3))
    assert sampler.ndim == 1

    X = X.reshape(-1, 2)
    sampler = Sampler()
    sampler.fit(X)
    assert sampler.ndim == 2
