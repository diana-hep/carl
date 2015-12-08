# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

from carl.distributions import DistributionMixin
from carl.distributions import Normal

import numpy as np
from numpy.testing import assert_equal, assert_raises
from nose.tools import assert_true

import theano
import theano.tensor as T
from theano.tensor import TensorVariable
from theano.tensor.sharedvar import SharedVariable


def test_mixin_parameters():
    # Check raw parameters
    p = Normal(mu=0.0, sigma=1.0)
    assert_true(isinstance(p, DistributionMixin))
    assert_equal(len(p.parameters_), 2)
    assert_true(p.mu in p.parameters_)
    assert_true(p.sigma in p.parameters_)
    assert_true(isinstance(p.mu, SharedVariable))
    assert_true(isinstance(p.sigma, SharedVariable))
    assert_equal(p.mu.get_value(), 0.0)
    assert_equal(p.sigma.get_value(), 1.0)
    assert_equal(len(p.observeds_), 1)
    assert_true(p.X in p.observeds_)
    assert_true(isinstance(p.X, TensorVariable))

    # Check external parameters
    mu = theano.shared(0.0)
    sigma = theano.shared(1.0)
    p = Normal(mu=mu, sigma=sigma)
    assert_equal(mu, p.mu)
    assert_equal(sigma, p.sigma)

    # Check composed expressions as parameters
    a = theano.shared(0.0)
    b = theano.shared(-1.0)
    mu = a + b - 1.0
    sigma = T.abs_(a * b)
    p = Normal(mu=mu, sigma=sigma)
    assert_true(a in p.parameters_)
    assert_true(b in p.parameters_)

    # Check with constants
    mu = T.constant(0.0)
    sigma = T.constant(1.0)
    p = Normal(mu=mu, sigma=sigma)
    assert_equal(len(p.parameters_), 0)
    assert_equal(len(p.constants_), 2)
    assert_true(mu in p.constants_)
    assert_true(sigma in p.constants_)

    # Compose parameters with observed variables
    a = theano.shared(1.0)
    b = theano.shared(0.0)
    y = T.dvector(name="y")
    p = Normal(mu=a * y + b)
    assert_equal(len(p.parameters_), 3)
    assert_true(a in p.parameters_)
    assert_true(b in p.parameters_)
    assert_equal(len(p.observeds_), 2)
    assert_true(y in p.observeds_)


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
