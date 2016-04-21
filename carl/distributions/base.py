"""This module implements commons for distributions."""

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
    """Check, convert and extract inputs of a parameter value.

    This function wraps scalar or lists into a Theano shared variable, then
    acting as a parameter. Theano expressions are left unchanged.

    Parameters
    ----------
    * `name` [string]:
        The parameter name.

    * `value` [theano expression, list or scalar]:
        The parameter value.

    Returns
    -------
    * `value` [theano expression]:
        The parameter expression.

    * `parameters` [set of theano shared variables]:
        Set of base shared variables on which `value` depends.

    * `constants` [set of theano constants]:
        Set of base constants on which `value` depends.

    * `observeds` [set of theano tensor variables]:
        Set of base unset variables on which `value` depends.
    """
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
        if isinstance(value, list):
            value = np.ndarray(value)

        if isinstance(value, np.ndarray):
            value = theano.shared(value, name=name)
        else:
            value = theano.shared(float(value), name=name)

        parameters.add(value)

    return value, parameters, constants, observeds


def bound(expression, out, *predicates):
    """Bound a theano expression.

    Parameters
    ----------
    * `expression` [theano expression]:
        The expression to bound.

    * `out` [theano expression]:
        The out-of-bounds value.

    * `*predicates` [list of theano expressions]:
        The list of predicates defining the boundaries of `expression`.

    Returns
    -------
    * `value` [theano expression]:
         The bounded expression.
    """
    guard = 1

    for p in predicates:
        guard *= p

    return T.switch(guard, expression, out)


class DistributionMixin(BaseEstimator):
    """Distribution mixin.

    This class defines the common API for distribution objects.
    """

    def pdf(self, X, **kwargs):
        """Probability density function.

        This function returns the value of the probability density function
        `p(x_i)`, at all `x_i` in `X`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `pdf` [array, shape=(n_samples,)]:
            `p(x_i)` for all `x_i` in `X`.
        """
        raise NotImplementedError

    def nll(self, X, **kwargs):
        """Negative log-likelihood.

        This function returns the value of the negative log-likelihood
        `- log(p(x_i))`, at all `x_i` in `X`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `nll` [array, shape=(n_samples,)]:
            `-log(p(x_i))` for all `x_i` in `X`.
        """
        raise NotImplementedError

    def cdf(self, X, **kwargs):
        """Cumulative distribution function.

        This function returns the value of the cumulative distribution function
        `F(x_i) = p(x <= x_i)`, at all `x_i` in `X`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `cdf` [array, shape=(n_samples,)]:
            `cdf(x_i)` for all `x_i` in `X`.
        """
        raise NotImplementedError

    def ppf(self, X, **kwargs):
        """Percent point function (inverse of cdf).

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `ppf` [array, shape=(n_samples,)]:
            Percent point function for all `x_i` in `X`.
        """
        raise NotImplementedError

    def rvs(self, n_samples, random_state=None, **kwargs):
        """Draw samples.

        Parameters
        ----------
        * `n_samples` [int]:
            The number of samples to draw.

        * `random_state` [integer or RandomState object]:
            The random seed.

        Returns
        -------
        * `X` [array, shape=(n_samples, n_features)]:
            The generated samples.
        """
        raise NotImplementedError

    def fit(self, X, **kwargs):
        """Fit the distribution parameters to data.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
        return self

    def score(self, X, **kwargs):
        """Evaluate the goodness of fit of the distribution w.r.t. `X`.

        The higher, the better.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `score` [float]:
            The goodness of fit.
        """
        return NotImplementedError

    @property
    def ndim(self):
        """The number of dimensions (or features) of the data."""
        return 1


class TheanoDistribution(DistributionMixin):
    """Base class for Theano-based distributions.

    Exposed attributes include:

    - `parameters_`: the set of parameters on which the distribution depends.

    - `constants_`: the set of constants on which the distribution depends.

    - `observeds_`: the set of unset variables on which the distribution
      depends. Values for these variables are required to be set using the
      `**kwargs` argument.
    """

    # Mixin interface
    X = T.dmatrix(name="X")  # Input expression is shared by all distributions
    p = T.dmatrix(name="p")

    def __init__(self, **parameters):
        """Constructor.

        Parameters
        ----------
        * `**parameters` [pairs of name/value]:
            The list of parameter names with their values.
        """
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

    def _make(self, expression, name, args=None, kwargs=None):
        if args is None:
            args = [self.X]

        if kwargs is None:
            kwargs = self.observeds_

        func = theano.function(
            [theano.In(v, name=v.name) for v in args] +
            [theano.In(v, name=v.name) for v in kwargs],
            expression,
            allow_input_downcast=True
        )

        setattr(self, name, func)

    def rvs(self, n_samples, random_state=None, **kwargs):
        rng = check_random_state(random_state)
        p = rng.rand(n_samples, 1)
        return self.ppf(p, **kwargs)

    # Scikit-Learn estimator interface
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

    def fit(self, X, bounds=None, constraints=None, use_gradient=True,
            optimizer=None, **kwargs):
        """Fit the distribution parameters to data by minimizing the negative
        log-likelihood of the data.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        * `bounds` [list of (parameter, (low, high))]:
            The parameter bounds.

        * `constraints`:
            The constraints on the parameters.

        * `use_gradient` [boolean, default=True]:
            Whether to use exact gradients (if `True`) or numerical gradients
            (if `False`).

        * `optimizer` [string]:
            The optimization method.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
        # Map parameters to placeholders
        param_to_placeholder = []
        param_to_index = {}

        for i, v in enumerate(self.parameters_):
            w = T.TensorVariable(v.type)
            param_to_placeholder.append((v, w))
            param_to_index[v] = i

        # Build bounds
        mapped_bounds = None

        if bounds is not None:
            mapped_bounds = [(None, None) for v in param_to_placeholder]

            for b in bounds:
                mapped_bounds[param_to_index[b["param"]]] = b["bounds"]

        # Build constraints
        mapped_constraints = None

        if constraints is not None:
            mapped_constraints = []

            for c in constraints:
                args = c["param"]
                if isinstance(args, SharedVariable):
                    args = (args, )

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
            [theano.In(v, name=v.name) for v in self.observeds_],
            T.sum(self.nll_),
            givens=param_to_placeholder,
            allow_input_downcast=True)

        def objective(x):
            return objective_(X, *x, **kwargs) / len(X)

        if use_gradient:
            gradient_ = theano.function(
                [self.X] + [w for _, w in param_to_placeholder] +
                [theano.In(v, name=v.name) for v in self.observeds_],
                theano.grad(T.sum(self.nll_),
                            [v for v, _ in param_to_placeholder]),
                givens=param_to_placeholder,
                allow_input_downcast=True)

            def gradient(x):
                return np.array(gradient_(X, *x, **kwargs)) / len(X)

        # Solve!
        x0 = np.array([v.get_value() for v, _ in param_to_placeholder])
        r = minimize(objective,
                     jac=gradient if use_gradient else None,
                     x0=x0,
                     method=optimizer,
                     bounds=mapped_bounds,
                     constraints=mapped_constraints)

        if r.success:
            # Assign the solution
            for i, value in enumerate(r.x):
                param_to_placeholder[i][0].set_value(value)

        else:
            print("Parameter fitting failed!")
            print(r)

        return self

    def score(self, X, **kwargs):
        """Evaluate the log-likelihood `-self.nll(X).sum()` of `X`.

        The higher, the better.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `score` [float]:
            The log-likelihood of `X`.
        """
        return -self.nll(X, **kwargs).sum()
