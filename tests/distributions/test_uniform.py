# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

from numpy.testing import assert_array_almost_equal
import scipy.stats as st
from sklearn.utils import check_random_state
import theano
import theano.tensor as T

from carl.distributions import Uniform


def test_uniform():
    rng = check_random_state(1)

    p_carl = Uniform(low=0.0, high=1.0)
    p_scipy = st.uniform(loc=0.0, scale=1.0)
    X =  3 * rng.rand(10, 1) - 1

    assert_array_almost_equal(p_carl.pdf(X).ravel(),
                              p_scipy.pdf(X.ravel()))
    assert_array_almost_equal(p_carl.cdf(X).ravel(),
                              p_scipy.cdf(X.ravel()))
