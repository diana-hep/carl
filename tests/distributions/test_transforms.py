# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from numpy.testing import assert_array_equal

from carl.distributions import Normal
from carl.distributions import MultivariateNormal
from carl.distributions import LinearTransform


def test_linear_transform_1d():
    p0 = Normal()
    pt = LinearTransform(p0, A=np.array([[0.5]]))

    X0 = p0.rvs(10, random_state=0)
    Xt = pt.rvs(10, random_state=0)

    assert X0.shape == Xt.shape
    assert_array_equal(X0 * 0.5, Xt)
    assert_array_equal(p0.pdf(X0), pt.pdf(Xt))
    assert_array_equal(p0.nll(X0), pt.nll(Xt))


def test_linear_transform_2d():
    p0 = MultivariateNormal(mu=np.array([0., 1.]), sigma=np.eye(2))
    pt = LinearTransform(p0, A=np.array([[1.0, 0],
                                         [0., 0.5]]))

    X0 = p0.rvs(10, random_state=0)
    Xt = pt.rvs(10, random_state=0)

    assert X0.shape == Xt.shape
    assert_array_equal(p0.pdf(X0), pt.pdf(Xt))
    assert_array_equal(p0.nll(X0), pt.nll(Xt))
