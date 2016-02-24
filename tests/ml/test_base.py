# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeRegressor

from carl.ml import as_classifier


def test_as_classifier():
    X, y = make_moons(n_samples=100, random_state=1)
    y = 2 * y - 1  # use -1/+1 labels

    clf = as_classifier(DecisionTreeRegressor())
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    predictions = clf.predict(X)

    assert_array_equal(probas.shape, (len(X), 2))
    assert_array_equal(predictions, y)

    y[-1] = 2
    clf = as_classifier(DecisionTreeRegressor())
    assert_raises(ValueError, clf.fit, X, y)
