# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from .base import DensityRatioMixin


class WrapAsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        return self

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


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
