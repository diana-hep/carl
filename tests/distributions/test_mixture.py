# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import scipy.stats as st
import theano
import theano.tensor as T

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal
from sklearn.utils import check_random_state

from carl.distributions import Normal
from carl.distributions import Exponential
from carl.distributions import Mixture


def test_mixture_api():
    p1 = Normal(mu=0.0, sigma=T.constant(1.0))
    p2 = Normal(mu=1.0, sigma=2.0)
    m = Mixture(components=[p1, p2], weights=[0.25])

    assert len(m.components) == 2
    assert len(m.weights) == 2

    assert len(m.parameters_) == 4
    assert len(m.constants_) == 1
    assert len(m.observeds_) == 0

    assert p1.mu in m.parameters_
    assert p1.sigma in m.constants_
    assert p2.mu in m.parameters_
    assert p2.sigma in m.parameters_
    assert m.X == p1.X
    assert m.X == p2.X

    m = Mixture(components=[p1, p2])
    assert m.weights[0].eval() == 0.5
    assert m.weights[1].eval() == 0.5

    y = T.dscalar(name="y")
    w1 = T.constant(0.25)
    w2 = y * 2
    m = Mixture(components=[p1, p2], weights=[w1, w2])

    assert_raises(ValueError, Mixture, components=[p1, p1, p1], weights=[1.0])


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
                              (w1 if w1 is not None
                                  else (1 - w0)) * q2.pdf(X).ravel())


def test_mixture_pdf():
    for w0, w1, mu1, sigma1, mu2, sigma2 in [(0.5, 0.5, 0.0, 1.0, 1.0, 2.0),
                                             (0.5, None, 0.0, 1.0, 1.0, 2.0),
                                             (0.1, 0.9, 1.0, 2.0, -1.0, 2.0)]:
        yield check_mixture_pdf, w0, w1, mu1, sigma1, mu2, sigma2


def test_rvs():
    p1 = Normal(mu=0.0, sigma=T.constant(1.0))
    p2 = Normal(mu=2.0, sigma=2.0)
    m = Mixture(components=[p1, p2], weights=[0.25])
    X = m.rvs(2000)
    assert (np.mean(X) - (0.25 * p1.mu.eval() + 0.75 * p2.mu.eval())) < 0.1


def test_fit():
    p1 = Normal(mu=T.constant(0.0), sigma=T.constant(2.0))
    p2 = Normal(mu=T.constant(3.0), sigma=T.constant(2.0))
    p3 = Exponential(inv_scale=T.constant(0.5))
    g = theano.shared(0.5)
    m = Mixture(components=[p1, p2, p3], weights=[g, g*g])

    X = np.concatenate([st.norm(loc=0.0, scale=2.0).rvs(300, random_state=0),
                        st.norm(loc=3.0, scale=2.0).rvs(100, random_state=1),
                        st.expon(scale=1. / 0.5).rvs(500, random_state=2)])
    X = X.reshape(-1, 1)
    s0 = m.score(X)

    m.fit(X)
    assert np.abs(g.eval() - 1. / 3.) < 0.05
    assert m.score(X) <= s0
