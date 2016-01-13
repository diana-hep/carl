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
from sklearn.utils import check_random_state

from theano.gof import graph
from theano.tensor.sharedvar import SharedVariable


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


def bound(expression, out, *predicates):
    guard = 1

    for p in predicates:
        guard *= p

    return T.switch(guard, expression, out)


class DistributionMixin(BaseEstimator):
    # Mixin interface
    def __init__(self, random_state=None):
        self.random_state = random_state

    def pdf(self, X, **kwargs):
        raise NotImplementedError

    def nnlf(self, X, **kwargs):
        raise NotImplementedError

    def cdf(self, X, **kwargs):
        raise NotImplementedError

    def ppf(self, X, **kwargs):
        raise NotImplementedError

    def fit(self, X, y=None, **kwargs):
        return self

    def score(self, X, **kwargs):
        return NotImplementedError


class TheanoDistribution(DistributionMixin):
    # Mixin interface
    X = T.dmatrix(name="X")  # Input expression is shared by all distributions
    p = T.dmatrix(name="p")

    def __init__(self, random_state=None, optimizer=None, **parameters):
        # Settings
        super(TheanoDistribution, self).__init__(random_state=random_state)
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
        if args is None:
            args = [self.X]

        if kwargs is None:
            kwargs = self.observeds_

        func = theano.function(
            [theano.Param(v, name=v.name) for v in args] +
            [theano.Param(v, name=v.name) for v in kwargs],
            expression,
            allow_input_downcast=True
        )

        setattr(self, name, func)

    def rvs(self, n_samples, **kwargs):
        rng = check_random_state(self.random_state)
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

    def fit(self, X, y=None, constraints=None, **kwargs):
        # Map parameters to placeholders
        param_to_placeholder = []
        param_to_index = {}

        for i, v in enumerate(self.parameters_):
            w = T.TensorVariable(v.type)
            param_to_placeholder.append((v, w))
            param_to_index[v.name] = i

        # Build constraints
        mapped_constraints = None

        if constraints is not None:
            mapped_constraints = []

            for c in constraints:
                if isinstance(c["param"], str):
                    args = (c["param"], )
                else:
                    args = c["param"]  # constraints can span multiple params!

                m_c = {
                    "type": c["type"],
                    "fun": lambda x: c["fun"](*[x[param_to_index[a]]
                                                for a in args])
                }

                if "jac" in c:
                    m_c["jac"] = lambda x: c["jac"](*[x[param_to_index[a]]
                                                      for a in args])

                mapped_constraints.append(m_c)

        # Derive objective and gradient
        objective_ = theano.function(
            [self.X] + [w for _, w in param_to_placeholder] +
            [theano.Param(v, name=v.name) for v in self.observeds_],
            T.sum(self.nnlf_),
            givens=param_to_placeholder,
            allow_input_downcast=True)

        gradient_ = theano.function(
            [self.X] + [w for _, w in param_to_placeholder] +
            [theano.Param(v, name=v.name) for v in self.observeds_],
            theano.grad(T.sum(self.nnlf_),
                        [v for v, _ in param_to_placeholder]),
            givens=param_to_placeholder,
            allow_input_downcast=True)

        def objective(x):
            return objective_(X, *x, **kwargs) / len(X)

        def gradient(x):
            return np.array(gradient_(X, *x, **kwargs)) / len(X)

        # Solve!
        x0 = np.array([v.get_value() for v, _ in param_to_placeholder])
        r = minimize(objective, jac=gradient, x0=x0,
                     method=self.optimizer, constraints=mapped_constraints)

        # Assign the solution
        for i, value in enumerate(r.x):
            param_to_placeholder[i][0].set_value(value)

        return self

    def score(self, X, **kwargs):
        return self.nnlf(X, **kwargs).sum()
