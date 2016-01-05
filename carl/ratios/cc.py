# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.model_selection import check_cv

from ..distributions import KernelDensity
from ..distributions import Histogram
from ..utils import as_classifier
from .base import DensityRatioMixin

# XXX: write own check_cv so that it can take a float


class CalibratedClassifierRatio(BaseEstimator, DensityRatioMixin):
    def __init__(self, base_estimator, calibration="histogram", cv=None):
        self.base_estimator = base_estimator
        self.calibration = calibration
        self.cv = cv

    def _fit_X_y(self, X_clf, y_clf, X_cal, y_cal):
        clf = clone(self.base_estimator)

        if isinstance(clf, RegressorMixin):
            clf = as_classifier(clf)

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
        self.identity_ = (numerator is not None) and (numerator is denominator)

        if self.identity_:
            return self

        if (numerator is not None and denominator is not None and
              n_samples is not None):
            X = np.vstack((numerator.rvs(n_samples // 2),
                           denominator.rvs(n_samples // 2)))
            y = np.zeros(n_samples, dtype=np.int)
            y[n_samples // 2:] = 1

        elif X is not None and y is not None:
            pass  # Use given X and y

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
        if self.identity_:
            if log:
                return np.zeros(len(X))
            else:
                return np.ones(len(X))

        else:
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
