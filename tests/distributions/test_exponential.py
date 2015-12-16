# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import scipy.stats as st
import theano
import theano.tensor as T

from numpy.testing import assert_array_almost_equal
from sklearn.utils import check_random_state

from carl.distributions import Exponential


def check_exponential(inv_scale):
    rng = check_random_state(1)

    p_carl = Exponential(inv_scale=inv_scale)
    p_scipy = st.expon(scale=1. / inv_scale)
    X = rng.rand(50, 1)

    assert_array_almost_equal(p_carl.pdf(X),
                              p_scipy.pdf(X.ravel()))
    assert_array_almost_equal(p_carl.cdf(X),
                              p_scipy.cdf(X.ravel()))
    assert_array_almost_equal(-np.log(p_carl.pdf(X)),
                              p_carl.nnlf(X))


def test_exponential():
    for inv_scale in [1, 2, 5]:
        yield check_exponential, inv_scale


def check_rvs(inv_scale, random_state):
    p = Exponential(inv_scale=inv_scale, random_state=random_state)
    samples = p.rvs(1000)
    assert np.abs(np.mean(samples) - 1. / inv_scale) <= 0.05


def test_rvs():
    for inv_scale, random_state in [(1, 0), (1, 1),
                                    (2, 3), (0.5, 4)]:
        yield check_rvs, inv_scale, random_state


def check_fit(inv_scale):
    p = Exponential()
    X = st.expon(scale=1. / inv_scale).rvs(5000,
                                           random_state=0).reshape(-1, 1)
    p.fit(X)
    assert np.abs(p.inv_scale.get_value() - inv_scale) <= 0.1


def test_fit():
    for inv_scale in [1, 2, 5]:
        yield check_fit, inv_scale
