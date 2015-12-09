# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import scipy.stats as st
import theano
import theano.tensor as T

from nose.tools import assert_equal
from nose.tools import assert_less_equal
from nose.tools import assert_in
from numpy.testing import assert_array_almost_equal
from sklearn.utils import check_random_state

from carl.distributions import Normal
from carl.distributions import Mixture


def test_mixture():
    p1 = Normal(mu=0.0, sigma=1.0)
    p2 = Normal(mu=1.0, sigma=2.0)
    m = Mixture(components=[p1, p2])

    assert_equal(len(m.weights), 2)
    assert_equal(len(m.components), 2)

    assert_equal(len(m.parameters_), 5)
    assert_equal(len(m.constants_), 0)
    assert_equal(len(m.observeds_), 1)

    assert_in(p1.mu, m.parameters_)
    assert_in(p1.sigma, m.parameters_)
    assert_in(p2.mu, m.parameters_)
    assert_in(p2.sigma, m.parameters_)
    assert_equal(m.X, p1.X)
    assert_equal(m.X, p2.X)
