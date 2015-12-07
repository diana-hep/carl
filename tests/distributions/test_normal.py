# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy.stats as st
from sklearn.utils import check_random_state
import theano
import theano.tensor as T

from carl.distributions import Normal

np.random.seed(1)


def test_normal():
    rng = check_random_state(1)

    p_carl = Normal(mu=0.0, sigma=1.0)
    p_scipy = st.norm(loc=0.0, scale=1.0)
    X = rng.rand(10, 1)

    assert_array_almost_equal(p_carl.pdf(X).ravel(),
                              p_scipy.pdf(X.ravel()))
    assert_array_almost_equal(p_carl.cdf(X).ravel(),
                              p_scipy.cdf(X.ravel()))
