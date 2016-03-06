# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from sklearn.utils import check_random_state

from carl.distributions import Join
from carl.distributions import Normal
from carl.distributions import Histogram


def test_join():
    p = Join(components=[Normal(mu=0), Normal(mu=1), Normal(mu=2)])
    assert p.ndim == 3
    assert len(p.parameters_) == 6

    X = p.rvs(10000, random_state=1)
    assert X.shape == (10000, 3)
    assert np.abs(np.mean(X[:, 0]) - 0.) < 0.05
    assert np.abs(np.mean(X[:, 1]) - 1.) < 0.05
    assert np.abs(np.mean(X[:, 2]) - 2.) < 0.05
    assert_array_almost_equal(-np.log(p.pdf(X)), p.nnlf(X))


def test_join_non_theano():
    h0 = Histogram(interpolation="linear", bins=30)
    h1 = Histogram(interpolation="linear", bins=30)
    h2 = Histogram(interpolation="linear", bins=30)

    h0.fit(Normal(mu=0).rvs(10000, random_state=0))
    h1.fit(Normal(mu=1).rvs(10000, random_state=1))
    h2.fit(Normal(mu=2).rvs(10000, random_state=2))

    p = Join(components=[h0, h1, h2])
    assert p.ndim == 3
    assert len(p.parameters_) == 0

    X = p.rvs(10000, random_state=1)
    assert X.shape == (10000, 3)
    assert np.abs(np.mean(X[:, 0]) - 0.) < 0.05
    assert np.abs(np.mean(X[:, 1]) - 1.) < 0.05
    assert np.abs(np.mean(X[:, 2]) - 2.) < 0.05
    assert_array_almost_equal(-np.log(p.pdf(X)), p.nnlf(X))
