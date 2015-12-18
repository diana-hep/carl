# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelEncoder

from .base import DensityRatioMixin


class WrapAsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        # Check inputs
        X, y = check_X_y(X, y)

        # Convert y
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y).astype(np.float)

        if len(label_encoder.classes_) != 2:
            raise ValueError

        self.classes_ = label_encoder.classes_

        # Fit regressor
        self.regressor_ = clone(self.regressor).fit(X, y)

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


class CalibratedClassifierRatio(BaseEstimator, DensityRatioMixin):
    def __init__(self, classifier, numerator=None, denominator=None,
                 n_samples=None, calibration=None, cv=None, decompose=False):
        self.classifier = classifier
        self.numerator = numerator
        self.denominator = denominator
        self.calibration = calibration
        self.cv = cv
        self.decompose = decompose

    def fit(self, X=None, y=None, **kwargs):
        # for fold in cv
        #       train classifier on train fold
        #       fit calibrator1 on num
        #       fit calibrator2 on den
        return self

    def predict(self, X=None, **kwargs):
        raise NotImplementedError

    def score(self, X=None, y=None, **kwargs):
        raise NotImplementedError
