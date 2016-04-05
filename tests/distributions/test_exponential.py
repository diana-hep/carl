# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import scipy.stats as st

from numpy.testing import assert_array_almost_equal
from sklearn.utils import check_random_state

from carl.distributions import Exponential


def check_exponential(inverse_scale):
    rng = check_random_state(1)

    p_carl = Exponential(inverse_scale=inverse_scale)
    p_scipy = st.expon(scale=1. / inverse_scale)
    X = rng.rand(50, 1)

    assert_array_almost_equal(p_carl.pdf(X),
                              p_scipy.pdf(X.ravel()))
    assert_array_almost_equal(p_carl.cdf(X),
                              p_scipy.cdf(X.ravel()))
    assert_array_almost_equal(-np.log(p_carl.pdf(X)),
                              p_carl.nll(X))


def test_exponential():
    for inverse_scale in [1, 2, 5]:
        yield check_exponential, inverse_scale


def check_rvs(inverse_scale, random_state):
    p = Exponential(inverse_scale=inverse_scale)
    samples = p.rvs(1000, random_state=random_state)
    assert np.abs(np.mean(samples) - 1. / inverse_scale) <= 0.05


def test_rvs():
    for inverse_scale, random_state in [(1, 0), (1, 1),
                                        (2, 3), (0.5, 4)]:
        yield check_rvs, inverse_scale, random_state


def check_fit(inverse_scale):
    p = Exponential()
    X = st.expon(scale=1. / inverse_scale).rvs(5000,
                                               random_state=0).reshape(-1, 1)
    p.fit(X)
    assert np.abs(p.inverse_scale.get_value() - inverse_scale) <= 0.1


def test_fit():
    for inverse_scale in [1, 2, 5]:
        yield check_fit, inverse_scale
