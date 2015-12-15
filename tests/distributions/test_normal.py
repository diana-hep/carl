# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import scipy.stats as st
import theano
import theano.tensor as T

from nose.tools import assert_less_equal
from numpy.testing import assert_array_almost_equal
from sklearn.utils import check_random_state

from carl.distributions import Normal


def check_normal(mu, sigma):
    rng = check_random_state(1)

    p_carl = Normal(mu=mu, sigma=sigma)
    p_scipy = st.norm(loc=mu, scale=sigma)
    X = rng.rand(50, 1)

    assert_array_almost_equal(p_carl.pdf(X).ravel(),
                              p_scipy.pdf(X.ravel()))
    assert_array_almost_equal(p_carl.cdf(X).ravel(),
                              p_scipy.cdf(X.ravel()))
    assert_array_almost_equal(-np.log(p_carl.pdf(X)),
                              p_carl.nnlf(X))


def test_normal():
    for mu, sigma in [(0., 1.), (-1., 1.5), (3., 2.)]:
        yield check_normal, mu, sigma


def check_rvs(mu, sigma, random_state):
    p = Normal(mu=mu, sigma=sigma, random_state=random_state)
    samples = p.rvs(1000)
    assert_less_equal(np.abs(np.mean(samples) - mu), 0.05)
    assert_less_equal(np.abs(np.std(samples) - sigma), 0.05)


def test_rvs():
    for mu, sigma, random_state in [(0, 1, 0), (1, 1, 1),
                                    (2, 2, 3), (-1, 0.5, 4)]:
        yield check_rvs, mu, sigma, random_state


def check_fit(mu, sigma):
    p = Normal()
    X = st.norm(loc=mu, scale=sigma).rvs(5000, random_state=0).reshape(-1, 1)
    p.fit(X)
    assert_less_equal(np.abs(p.mu.get_value() - mu), 0.1)
    assert_less_equal(np.abs(p.sigma.get_value() - sigma), 0.1)


def test_fit():
    for mu, sigma in [(0., 1.), (-1., 1.5), (3., 2.)]:
        yield check_fit, mu, sigma
