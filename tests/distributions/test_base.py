# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import theano
import theano.tensor as T

from nose.tools import assert_equal
from nose.tools import assert_in
from nose.tools import assert_true
from numpy.testing import assert_raises
from theano.tensor import TensorVariable
from theano.tensor.sharedvar import SharedVariable

from carl.distributions import DistributionMixin
from carl.distributions import Normal


def test_mixin_base():
    # Check raw parameters
    p = Normal(mu=0.0, sigma=1.0)
    assert_true(isinstance(p, DistributionMixin))
    assert_equal(len(p.parameters_), 2)
    assert_in(p.mu, p.parameters_)
    assert_in(p.sigma, p.parameters_)
    assert_true(isinstance(p.mu, SharedVariable))
    assert_true(isinstance(p.sigma, SharedVariable))
    assert_equal(p.mu.get_value(), 0.0)
    assert_equal(p.sigma.get_value(), 1.0)
    assert_equal(len(p.observeds_), 1)
    assert_in(p.X, p.observeds_)
    assert_true(isinstance(p.X, TensorVariable))


def test_mixin_external():
    # Check external parameters
    mu = theano.shared(0.0)
    sigma = theano.shared(1.0)
    p = Normal(mu=mu, sigma=sigma)
    assert_equal(mu, p.mu)
    assert_equal(sigma, p.sigma)


def test_mixin_constants():
    # Check with constants
    mu = T.constant(0.0)
    sigma = T.constant(1.0)
    p = Normal(mu=mu, sigma=sigma)
    assert_equal(len(p.parameters_), 0)
    assert_equal(len(p.constants_), 2)
    assert_in(mu, p.constants_)
    assert_in(sigma, p.constants_)


def test_mixin_composition():
    # Check composed expressions as parameters
    a = theano.shared(0.0)
    b = theano.shared(-1.0)
    mu = a + b - 1.0
    sigma = T.abs_(a * b)
    p = Normal(mu=mu, sigma=sigma)
    assert_in(a, p.parameters_)
    assert_in(b, p.parameters_)

    # Compose parameters with observed variables
    a = theano.shared(1.0)
    b = theano.shared(0.0)
    y = T.dmatrix(name="y")
    p = Normal(mu=a * y + b)
    assert_equal(len(p.parameters_), 3)
    assert_in(a, p.parameters_)
    assert_in(b, p.parameters_)
    assert_in(p.sigma, p.parameters_)
    assert_true(p.mu not in p.parameters_)
    assert_equal(len(p.observeds_), 2)
    assert_in(y, p.observeds_)
    assert_in(p.X, p.observeds_)


def test_mixin_sklearn_params():
    # get_params
    p = Normal(mu=0.0, sigma=1.0)
    params = p.get_params()
    assert_equal(len(params), 3)
    assert_true("random_state" in params)
    assert_true("mu" in params)
    assert_true("sigma" in params)

    # set_params
    old_rng = p.get_params()["random_state"]
    assert_equal(old_rng, None)
    p.set_params(random_state=42)
    new_rng = p.get_params()["random_state"]
    assert_equal(new_rng, 42)

    # for parameters, set_params should change the value contained
    old_mu = p.get_params()["mu"]
    p.set_params(mu=42.0)
    new_mu = p.get_params()["mu"]
    assert_true(old_mu is new_mu)
    assert_equal(new_mu.get_value(), 42.0)
