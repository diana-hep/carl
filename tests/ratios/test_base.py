# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from numpy.testing import assert_array_almost_equal

from carl.distributions import Normal
from carl.distributions import Mixture

from carl.ratios import DensityRatioMixin
from carl.ratios import KnownDensityRatio
from carl.ratios import InverseRatio
from carl.ratios import DecomposedRatio
from carl.ratios import ClassifierRatio
from carl.learning import CalibratedClassifierCV

from sklearn.linear_model import ElasticNetCV


def test_known_density():
    components = [Normal(mu=0.0), Normal(mu=0.25), Normal(mu=0.5)]
    p0 = Mixture(components=components, weights=[0.45, 0.1, 0.45])
    p1 = Mixture(components=[components[0]] + [components[2]])

    ratio = KnownDensityRatio(numerator=p0, denominator=p1)

    reals = np.linspace(-0.5, 1.0, num=100).reshape(-1, 1)
    assert ratio.score(reals, p0.pdf(reals) / p1.pdf(reals)) > -0.01
    assert np.mean(np.abs(np.log(ratio.predict(reals)) -
                          ratio.predict(reals, log=True))) < 0.01


def check_inverse(constant):
    class Constant(DensityRatioMixin):
        def __init__(self, c):
            self.c = c

        def predict(self, X, log=False, **kwargs):
            if log:
                return np.log(np.ones(len(X)) * self.c)
            else:
                return np.ones(len(X)) * self.c

    ratio = InverseRatio(Constant(constant))
    X = np.zeros((5, 1))

    assert_array_almost_equal(np.ones(len(X)) / constant, ratio.predict(X))
    assert_array_almost_equal(np.log(np.ones(len(X)) / constant),
                              ratio.predict(X, log=True))


def test_inverse():
    for constant in [10.0, 5, -5]:
        yield check_inverse, constant


def test_decomposed_ratio():
    components = [Normal(mu=0.0), Normal(mu=0.25), Normal(mu=0.5)]
    p0 = Mixture(components=components, weights=[0.45, 0.1, 0.45])
    p1 = Mixture(components=[components[0]] + [components[2]])

    ratio = DecomposedRatio(
        ClassifierRatio(CalibratedClassifierCV(base_estimator=ElasticNetCV())))
    ratio.fit(numerator=p0, denominator=p1, n_samples=10000)

    reals = np.linspace(-0.5, 1.0, num=100).reshape(-1, 1)
    assert ratio.score(reals, p0.pdf(reals) / p1.pdf(reals)) > -0.1
    assert np.mean(np.abs(np.log(ratio.predict(reals)) -
                          ratio.predict(reals, log=True))) < 0.01


def test_decomposed_ratio_identity():
    components = [Normal(mu=0.0), Normal(mu=0.25), Normal(mu=0.5)]
    p = Mixture(components=components, weights=[0.45, 0.1, 0.45])

    ratio = DecomposedRatio(
        ClassifierRatio(CalibratedClassifierCV(base_estimator=ElasticNetCV())))
    ratio.fit(numerator=p, denominator=p, n_samples=10000)

    reals = np.linspace(-0.5, 1.0, num=100).reshape(-1, 1)
    assert ratio.score(reals, p.pdf(reals) / p.pdf(reals)) == 0.0
    assert_array_almost_equal(ratio.predict(reals), np.ones(len(reals)))
    assert_array_almost_equal(ratio.predict(reals, log=True),
                              np.zeros(len(reals)))
