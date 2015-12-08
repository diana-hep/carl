# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

import theano
import theano.tensor as T
from theano import Constant
from theano import Variable
from theano.gof import graph
from theano.tensor.sharedvar import SharedVariable

# ???: define the bounds of the parameters

def check_parameter(name, value):
    # Accept expressions only if they depend on bound variables
    if isinstance(value, Variable):
        inputs = graph.inputs([value])
        for var in inputs:
            if (not isinstance(var, SharedVariable) and
                not isinstance(var, Constant)):
                raise ValueError("Variable {} is free, or depends on a "
                                 "free variable.".format(value))

    # Accept constants
    elif isinstance(value, Constant):
        pass

    # Accept raw values, and store them in a shared variable
    else:
        value = theano.shared(value, name=name)

    return value


class DistributionMixin(BaseEstimator):
    def __init__(self, random_state=None, **parameters):
        # Settings
        self.random_state = random_state

        # Validate parameters of the distribution
        self.parameters_ = {}
        for name, value in parameters.items():
            setattr(self, name, check_parameter(name, value))
            self.parameters_[name] = getattr(self, name)

        # Default observed variable is a scalar
        self.X = T.dmatrix(name="X")
        self.observed_ = {"X": self.X}

    # Distribution interface
    def rvs(n_samples):
        raise NotImplementedError

    def pdf(self, X):
        raise NotImplementedError

    # Scikit-Learn estimator interface
    def fit(self, X, y=None):
        raise NotImplementedError

    def score(self, X):
        raise NotImplementedError

    def get_params(self, deep=True):
        return super(DistributionMixin, self).get_params(deep=deep)

    def set_params(self, **params):
        for name, value in params.items():
            if name in self.parameters_:
                getattr(self, name).set_value(value)
            else:
                super(DistributionMixin, self).set_params(**{name: value})
