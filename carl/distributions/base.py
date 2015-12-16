# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
import theano
import theano.tensor as T

from scipy.optimize import minimize

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state as check_random_state_sklearn

from theano.gof import graph
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.sharedvar import SharedVariable

# ???: define the bounds of the parameters

def check_parameter(name, value):
    parameters = set()
    constants = set()
    observeds = set()

    if isinstance(value, SharedVariable):
        parameters.add(value)
    elif isinstance(value, T.TensorConstant):
        constants.add(value)
    elif isinstance(value, T.TensorVariable):
        inputs = graph.inputs([value])

        for var in inputs:
            if isinstance(var, SharedVariable):
                parameters.add(var)
            elif isinstance(var, T.TensorConstant):
                constants.add(var)
            elif isinstance(var, T.TensorVariable):
                if not var.name:
                    raise ValueError("Observed variables must be named.")
                observeds.add(var)
    else:
        value = theano.shared(float(value), name=name)
        parameters.add(value)

    return value, parameters, constants, observeds


def check_random_state_theano(random_state):
    if isinstance(random_state, RandomStreams):
        return random_state
    elif isinstance(random_state, np.random.RandomState):
        random_state = random_state.randint(np.iinfo(np.int32).max)

    return RandomStreams(seed=random_state)


def bound(expression, out, *predicates):
    guard = 1
    for p in predicates:
        guard *= p

    return T.switch(guard, expression, out)


class DistributionMixin(BaseEstimator):
    # Mixin interface
    X = T.dmatrix(name="X")  # Input expression is shared by all distributions
    p = T.dmatrix(name="p")

    def __init__(self, random_state=None, optimizer=None, **parameters):
        # Settings
        self.random_state = random_state
        self.optimizer = optimizer

        # Validate parameters of the distribution
        self.parameters_ = set()        # base parameters
        self.constants_ = set()         # base constants
        self.observeds_ = set()         # base observeds

        # XXX: check that no two variables in observeds_ have the same name

        for name, value in parameters.items():
            v, p, c, o = check_parameter(name, value)
            setattr(self, name, v)

            for p_i in p:
                self.parameters_.add(p_i)
            for c_i in c:
                self.constants_.add(c_i)
            for o_i in o:
                self.observeds_.add(o_i)

    def make_(self, expression, name, args=None, kwargs=None):
        if hasattr(self, name):
            raise ValueError("Attribute {} already exists!")

        if args is None:
            args = [self.X]

        if kwargs is None:
            kwargs = self.observeds_

        func = theano.function(
            args + [theano.Param(v, name=v.name) for v in kwargs],
            expression,
            allow_input_downcast=True
        )

        setattr(self, name, func)

    def rvs(self, n_samples, **kwargs):
        rng = check_random_state_sklearn(self.random_state)
        p = rng.rand(n_samples, 1)
        return self.ppf(p, **kwargs)

    # Scikit-Learn estimator interface
    def get_params(self, deep=True):
        return super(DistributionMixin, self).get_params(deep=deep)

    def set_params(self, **params):
        for name, value in params.items():
            var = getattr(self, name, None)

            if isinstance(var, SharedVariable):
                var.set_value(value)
            elif (isinstance(var, T.TensorVariable) or
                  isinstance(var, T.TensorConstant)):
                raise ValueError("Only shared variables can be updated.")
            else:
                super(DistributionMixin, self).set_params(**{name: value})

        # XXX: shall we also allow replacement of variables and
        #      recompile all expressions instead?

    def fit(self, X, **kwargs):
        shared_to_symbols = []
        for v in self.parameters_:
            w = T.TensorVariable(v.type)
            shared_to_symbols.append((v, w))

        objective_ = theano.function(
            [self.X] +
                [w for _, w in shared_to_symbols] +
                [theano.Param(v, name=v.name)
                     for v in self.observeds_ if v is not self.X],
            T.sum(self.nnlf_),
            givens=shared_to_symbols,
            allow_input_downcast=True)

        gradient_ = theano.function(
            [self.X] +
                [w for _, w in shared_to_symbols] +
                [theano.Param(v, name=v.name)
                     for v in self.observeds_ if v is not self.X],
            theano.grad(T.sum(self.nnlf_), [v for v, _ in shared_to_symbols]),
            givens=shared_to_symbols,
            allow_input_downcast=True)

        def objective(x):
            return objective_(X, *x, **kwargs) / len(X)

        def gradient(x):
            return np.array(gradient_(X, *x, **kwargs)) / len(X)

        x0 = np.array([v.get_value() for v, _ in shared_to_symbols])
        r = minimize(objective, jac=gradient, x0=x0, method=self.optimizer)

        for i, value in enumerate(r.x):
            shared_to_symbols[i][0].set_value(value)

        return self

    def score(self, X, **kwargs):
        return self.nnlf(X, **kwargs).sum()
