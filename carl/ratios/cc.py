# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.model_selection import check_cv
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelEncoder

from ..distributions import KernelDensity
from ..distributions import Histogram
from .base import DensityRatioMixin

# XXX: depending on the calibration algorithm, it might be better to fit
#      on decision_function rather than on predict_proba
# XXX: implement decomposition in case of mixtures
# XXX: write own check_cv so that it can take a float


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
    def __init__(self, base_estimator, calibration="histogram", cv=None,
                 decompose=False):
        self.base_estimator = base_estimator
        self.calibration = calibration
        self.cv = cv
        self.decompose = decompose

    def _fit_X_y(self, X_clf, y_clf, X_cal, y_cal):
        clf = clone(self.base_estimator)

        if isinstance(clf, RegressorMixin):
            clf = WrapAsClassifier(clf)

        clf.fit(X_clf, y_clf)

        if self.calibration == "kde":
            cal_num = KernelDensity()
            cal_den = KernelDensity()

        elif self.calibration == "histogram":
            cal_num = Histogram(bins=100, range=[(0.0, 1.0)])
            cal_den = Histogram(bins=100, range=[(0.0, 1.0)])

        else:
            cal_num = clone(self.calibration)
            cal_den = clone(self.calibration)

        X_num = clf.predict_proba(X_cal[y_cal == 0])[:, 0]
        X_den = clf.predict_proba(X_cal[y_cal == 1])[:, 0]
        cal_num.fit(X_num.reshape(-1, 1))
        cal_den.fit(X_den.reshape(-1, 1))

        return clf, cal_num, cal_den

    def fit(self, X=None, y=None, numerator=None, denominator=None,
            n_samples=None, **kwargs):
        if X is not None and y is not None:
            pass  # use given X and y
        elif (numerator is not None and denominator is not None and
              n_samples is not None):
            X = np.vstack((numerator.rvs(n_samples // 2),
                           denominator.rvs(n_samples // 2)))
            y = np.zeros(n_samples, dtype=np.int)
            y[n_samples // 2:] = 1
        else:
            raise ValueError

        self.classifiers_ = []
        self.calibrators_ = []

        cv = check_cv(self.cv, y, classifier=True)

        for train, calibrate in cv.split(X, y):
            clf, cal_num, cal_den = self._fit_X_y(X[train], y[train],
                                                  X[calibrate], y[calibrate])
            self.classifiers_.append(clf)
            self.calibrators_.append((cal_num, cal_den))

        return self

    def predict(self, X, log=False, **kwargs):
        r = np.zeros(len(X))

        for clf, (cal_num, cal_den) in zip(self.classifiers_,
                                           self.calibrators_):
            p = clf.predict_proba(X)[:, 0].reshape(-1, 1)

            if log:
                r += -cal_num.nnlf(p) + cal_den.nnlf(p)
            else:
                r += cal_num.pdf(p) / cal_den.pdf(p)

        return r / len(self.classifiers_)

    def score(self, X, y, **kwargs):
        raise NotImplementedError
