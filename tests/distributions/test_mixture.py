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


def test_mixture_api():
    p1 = Normal(mu=0.0, sigma=T.constant(1.0))
    p2 = Normal(mu=1.0, sigma=2.0)
    m = Mixture(components=[p1, p2], weights=[0.25])

    assert_equal(len(m.components), 2)
    assert_equal(len(m.weights), 2)

    assert_equal(len(m.parameters_), 4)
    assert_equal(len(m.constants_), 1)
    assert_equal(len(m.observeds_), 1)

    assert_in(p1.mu, m.parameters_)
    assert_in(p1.sigma, m.constants_)
    assert_in(p2.mu, m.parameters_)
    assert_in(p2.sigma, m.parameters_)
    assert_equal(m.X, p1.X)
    assert_equal(m.X, p2.X)


def check_mixture_pdf(w0, w1, mu1, sigma1, mu2, sigma2):
    rng = check_random_state(1)

    p1 = Normal(mu=mu1, sigma=sigma1)
    p2 = Normal(mu=mu2, sigma=sigma2)
    m = Mixture(components=[p1, p2], weights=[w0, w1])
    q1 = st.norm(loc=mu1, scale=sigma1)
    q2 = st.norm(loc=mu2, scale=sigma2)

    X = rng.rand(50, 1)
    assert_array_almost_equal(m.pdf(X).ravel(),
                              w0 * q1.pdf(X).ravel() +
                              (w1 if w1 is not None else (1 - w0)) * q2.pdf(X).ravel())


def test_mixture_pdf():
    for w0, w1, mu1, sigma1, mu2, sigma2 in [(0.5, 0.5, 0.0, 1.0, 1.0, 2.0),
                                             (0.5, None, 0.0, 1.0, 1.0, 2.0),
                                             (0.1, 0.9, 1.0, 2.0, -1.0, 2.0)]:
        yield check_mixture_pdf, w0, w1, mu1, sigma1, mu2, sigma2
