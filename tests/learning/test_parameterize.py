# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_raises

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

import theano

from carl.distributions import Normal
from carl.learning import ParameterStacker
from carl.learning import ParameterizedClassifier
from carl.learning import ParameterizedRegressor
from carl.learning import make_parameterized_classification


def test_parameter_stacker():
    mu = theano.shared(0)
    sigma = theano.shared(1)
    p = Normal(mu=mu, sigma=sigma)
    X = p.rvs(10)

    tf = ParameterStacker(params=[mu, sigma])
    Xt = tf.transform(X)
    assert Xt.shape == (10, 1+2)
    assert_array_almost_equal(Xt[:, 1], np.zeros(10))
    assert_array_almost_equal(Xt[:, 2], np.ones(10))

    mu.set_value(1)
    Xt = tf.transform(X)
    assert_array_almost_equal(Xt[:, 1], np.ones(10))


def test_parameterized_classifier():
    mu0 = theano.shared(0)
    mu1 = theano.shared(1)
    p0 = Normal(mu=mu0)
    p1 = Normal(mu=mu1)

    X, y = make_parameterized_classification(p0, p1, 100, [mu0, mu1])

    clf = ParameterizedClassifier(DecisionTreeClassifier(), params=[mu0, mu1])
    clf.fit(X, y)

    assert clf.n_features_ == 1
    assert_array_almost_equal(y, clf.predict(X))


def test_parameterized_regressor():
    mu = theano.shared(0)
    p = Normal(mu=mu)

    X = p.rvs(100)
    y = p.pdf(X).astype(np.float32)

    tf = ParameterStacker(params=[mu])
    clf = ParameterizedRegressor(DecisionTreeRegressor(), params=[mu])
    clf.fit(tf.transform(X), y)

    assert clf.n_features_ == 1
    assert_array_almost_equal(y, clf.predict(tf.transform(X)), decimal=3)


def test_make_parameterized_classification():
    # Simple case
    mu0 = theano.shared(0.)
    mu1 = theano.shared(1.)
    p0 = Normal(mu=mu0)
    p1 = Normal(mu=mu1)
    X, y = make_parameterized_classification(p0, p1, 100, [mu0, mu1])

    assert X.shape == (100, 1+2)
    assert_array_almost_equal(X[:, 1], np.zeros(100))
    assert_array_almost_equal(X[:, 2], np.ones(100))

    # Grid of parameter values
    X, y = make_parameterized_classification(p0, p1, 100,
                                             [(mu0, [0, 0.5]),
                                              (mu1, [0.5, 1.5])])

    assert X.shape == (100, 1+2)
    assert_array_equal(np.unique(X[:, 1]), [0, 0.5])
    assert_array_equal(np.unique(X[:, 2]), [0.5, 1.5])

    d = set()
    for row in X[:, 1:]:
        d.add(tuple(row))

    assert_array_equal(np.array(sorted(d)), [[0., 0.5], [0., 1.5],
                                             [0.5, 0.5], [0.5, 1.5]])
