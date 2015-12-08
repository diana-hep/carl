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
    for mu, sigma in [(0., 1.), (1., 1.), (0., 2.)]:
        yield check_normal, mu, sigma
