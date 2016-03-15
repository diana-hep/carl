# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_raises
from sklearn.utils.testing import assert_greater

from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import brier_score_loss

from carl.learning import CalibratedClassifierCV


def check_calibration(method):
    # Adpated from sklearn/tests/test_calibration.py
    # Authors: Alexandre Gramfort
    # License: BSD 3 clause

    n_samples = 100
    X, y = make_classification(n_samples=2 * n_samples, n_features=6,
                               random_state=42)

    X -= X.min()  # MultinomialNB only allows positive X

    # split train and test
    X_train, y_train = X[:n_samples], y[:n_samples]
    X_test, y_test = X[n_samples:], y[n_samples:]

    # Naive-Bayes
    clf = MultinomialNB().fit(X_train, y_train)
    prob_pos_clf = clf.predict_proba(X_test)[:, 1]

    pc_clf = CalibratedClassifierCV(clf, cv=y.size + 1)
    assert_raises(ValueError, pc_clf.fit, X, y)

    pc_clf = CalibratedClassifierCV(clf, method=method, cv=2)
    # Note that this fit overwrites the fit on the entire training set
    pc_clf.fit(X_train, y_train)
    prob_pos_pc_clf = pc_clf.predict_proba(X_test)[:, 1]

    # Check that brier score has improved after calibration
    assert_greater(brier_score_loss(y_test, prob_pos_clf),
                   brier_score_loss(y_test, prob_pos_pc_clf))

    # Check invariance against relabeling [0, 1] -> [1, 2]
    pc_clf.fit(X_train, y_train + 1)
    prob_pos_pc_clf_relabeled = pc_clf.predict_proba(X_test)[:, 1]
    assert_array_almost_equal(prob_pos_pc_clf,
                              prob_pos_pc_clf_relabeled)

    # Check invariance against relabeling [0, 1] -> [-1, 1]
    pc_clf.fit(X_train, 2 * y_train - 1)
    prob_pos_pc_clf_relabeled = pc_clf.predict_proba(X_test)[:, 1]
    assert_array_almost_equal(prob_pos_pc_clf,
                              prob_pos_pc_clf_relabeled)

    # Check invariance against relabeling [0, 1] -> [1, 0]
    pc_clf.fit(X_train, (y_train + 1) % 2)
    prob_pos_pc_clf_relabeled = pc_clf.predict_proba(X_test)[:, 1]
    if method == "sigmoid":
        assert_array_almost_equal(prob_pos_pc_clf,
                                  1 - prob_pos_pc_clf_relabeled)
    else:
        # Isotonic calibration is not invariant against relabeling
        # but should improve in both cases
        assert_greater(brier_score_loss(y_test, prob_pos_clf),
                       brier_score_loss((y_test + 1) % 2,
                                        prob_pos_pc_clf_relabeled))


def test_calibration():
    for method in ["isotonic", "sigmoid", "histogram", "kde", "interpolated-isotonic"]:
        yield check_calibration, method
