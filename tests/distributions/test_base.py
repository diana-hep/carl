# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import inspect
import numpy as np
import theano
import theano.tensor as T

from numpy.testing import assert_raises
from theano.tensor import TensorVariable
from theano.tensor.sharedvar import SharedVariable

from carl.distributions import DistributionMixin
from carl.distributions import Normal


def test_mixin_base():
    # Check raw parameters
    p = Normal(mu=0.0, sigma=1.0)
    assert isinstance(p, DistributionMixin)
    assert len(p.parameters_) == 2
    assert p.mu in p.parameters_
    assert p.sigma in p.parameters_
    assert isinstance(p.mu, SharedVariable)
    assert isinstance(p.sigma, SharedVariable)
    assert p.mu.get_value() == 0.0
    assert p.sigma.get_value() == 1.0
    assert len(p.observeds_) == 0
    assert isinstance(p.X, TensorVariable)


def test_mixin_external():
    # Check external parameters
    mu = theano.shared(0.0)
    sigma = theano.shared(1.0)
    p = Normal(mu=mu, sigma=sigma)
    assert mu == p.mu
    assert sigma == p.sigma


def test_mixin_constants():
    # Check with constants
    mu = T.constant(0.0)
    sigma = T.constant(1.0)
    p = Normal(mu=mu, sigma=sigma)
    assert len(p.parameters_) == 0
    assert len(p.constants_) == 2
    assert mu in p.constants_
    assert sigma in p.constants_


def test_mixin_composition():
    # Check composed expressions as parameters
    a = theano.shared(0.0)
    b = theano.shared(-1.0)
    mu = a + b - 1.0
    sigma = T.abs_(a * b)
    p = Normal(mu=mu, sigma=sigma)
    assert a in p.parameters_
    assert b in p.parameters_

    # Compose parameters with observed variables
    a = theano.shared(1.0)
    b = theano.shared(0.0)
    y = T.dmatrix(name="y")
    p = Normal(mu=a * y + b)
    assert len(p.parameters_) == 3
    assert a in p.parameters_
    assert b in p.parameters_
    assert p.sigma in p.parameters_
    assert p.mu not in p.parameters_
    assert len(p.observeds_) == 1
    assert y in p.observeds_

    # Check signatures
    data_X = np.random.rand(10, 1)
    data_y = np.random.rand(10, 1)
    p.pdf(X=data_X, y=data_y)
    p.cdf(X=data_X, y=data_y)
    p.rvs(10, y=data_y)

    # Check error
    a = theano.shared(1.0)
    b = theano.shared(0.0)
    y = T.dmatrix()  # y must be named
    assert_raises(ValueError, Normal, mu=a * y + b)


def test_mixin_sklearn_params():
    # get_params
    p = Normal(mu=0.0, sigma=1.0)
    params = p.get_params()
    assert len(params) == 2
    assert "mu" in params
    assert "sigma" in params

    # for parameters, set_params should change the value contained
    old_mu = p.get_params()["mu"]
    p.set_params(mu=42.0)
    new_mu = p.get_params()["mu"]
    assert old_mu is new_mu
    assert new_mu.get_value() == 42.0

    # check errors
    p = Normal(mu=T.constant(0.0), sigma=1.0)
    assert_raises(ValueError, p.set_params, mu=1.0)
