# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

import sklearn
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils import check_X_y


def as_classifier(regressor):
    class Wrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, base_estimator):
            self.base_estimator = base_estimator

        def fit(self, X, y, **kwargs):
            # Check inputs
            X, y = check_X_y(X, y)

            # Convert y
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y).astype(np.float)

            if len(label_encoder.classes_) != 2:
                raise ValueError

            self.classes_ = label_encoder.classes_

            # Fit regressor
            self.regressor_ = clone(self.base_estimator).fit(X, y, **kwargs)

            return self

        def predict(self, X):
            return np.where(self.predict_proba(X)[:, 1] >= 0.5,
                            self.classes_[1],
                            self.classes_[0])

        def predict_proba(self, X):
            X = check_array(X)

            p = self.regressor_.predict(X)
            p = np.clip(p, 0., 1.)
            probas = np.zeros((len(X), 2))
            probas[:, 0] = 1. - p
            probas[:, 1] = p

            return probas

        def score(self, X, y):
            return self.regressor_.score(X, y)

    return Wrapper(regressor)


if int(sklearn.__version__[2:4]) < 18:
    class _CVIterableWrapper:
        # Backport from sklearn.model_selection
        def __init__(self, cv):
            self.cv = cv

        def get_n_splits(self, X=None, y=None, labels=None):
            return len(self.cv)

        def split(self, X=None, y=None, labels=None):
            for train, test in self.cv:
                yield train, test


def check_cv(cv=3, X=None, y=None, classifier=False):
    if int(sklearn.__version__[2:4]) >= 18:
        from sklearn.model_selection import check_cv as sklearn_check_cv
        return sklearn_check_cv(cv, y=y, classifier=classifier)

    else:
        from sklearn.cross_validation import check_cv as sklearn_check_cv
        return _CVIterableWrapper(sklearn_check_cv(cv, X=X, y=y,
                                                   classifier=classifier))
