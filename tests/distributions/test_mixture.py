# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import scipy.stats as st
import theano
import theano.tensor as T
import types

from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from sklearn.utils import check_random_state

from carl.distributions import Normal
from carl.distributions import MultivariateNormal
from carl.distributions import Exponential
from carl.distributions import Histogram
from carl.distributions import Mixture


def test_mixture_api():
    # Check basic API
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
    assert m.ndim == p1.ndim
    assert m.ndim == p2.ndim

    m = Mixture(components=[p1, p2])
    w = m.compute_weights()
    assert_array_equal(w, [0.5, 0.5])

    y = T.dscalar(name="y")
    w1 = T.constant(0.25)
    w2 = y * 2
    m = Mixture(components=[p1, p2], weights=[w1, w2])
    assert y in m.observeds_

    # Check errors
    assert_raises(ValueError, Mixture,
                  components=[p1, p1, p1], weights=[1.0])


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
    X = m.rvs(2000, random_state=1)
    assert (np.mean(X) - (0.25 * p1.mu.eval() + 0.75 * p2.mu.eval())) < 0.1


def test_fit():
    p1 = Normal(mu=T.constant(0.0), sigma=T.constant(2.0))
    p2 = Normal(mu=T.constant(3.0), sigma=T.constant(2.0))
    p3 = Exponential(inverse_scale=T.constant(0.5))
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


def test_likelihood_free_mixture():
    p1 = Normal()
    p2 = Normal(mu=2.0)
    h1 = Histogram(bins=50).fit(p1.rvs(10000, random_state=0))
    h2 = Histogram(bins=50).fit(p2.rvs(10000, random_state=1))
    m1 = Mixture(components=[p1, p2])
    m2 = Mixture(components=[h1, h2])

    # Check whether pdf, nll and cdf have been overriden
    assert isinstance(m1.pdf, theano.compile.function_module.Function)
    assert isinstance(m1.nll, theano.compile.function_module.Function)
    assert isinstance(m1.cdf, theano.compile.function_module.Function)
    assert isinstance(m2.pdf, types.MethodType)
    assert isinstance(m2.nll, types.MethodType)
    assert isinstance(m2.cdf, types.MethodType)

    # Compare pdfs
    rng = check_random_state(1)
    X = rng.rand(100, 1) * 10 - 5
    assert np.mean(np.abs(m1.pdf(X) - m2.pdf(X))) < 0.05

    # Test sampling
    X = m2.rvs(10)
    assert X.shape == (10, 1)

    # Check errors
    assert_raises(NotImplementedError, m2.fit, X)


def test_mv_mixture():
    p1 = MultivariateNormal(mu=np.array([0.0, 0.0]),
                            sigma=np.eye(2))
    p2 = MultivariateNormal(mu=np.array([2.0, 2.0]),
                            sigma=0.5 * np.eye(2))
    m = Mixture(components=[p1, p2])
    assert m.ndim == 2
    X = m.rvs(100)
    assert X.shape == (100, 2)

    assert_raises(ValueError, Mixture, components=[p1, Normal()])
