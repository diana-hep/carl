# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.calibration import CalibratedClassifierCV as CCCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils import check_X_y

from ..distributions import KernelDensity
from ..distributions import Histogram
from .base import as_classifier
from .base import check_cv


class CalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, method="histogram", cv=1, bins="auto"):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.bins = bins

    def _fit_calibrators(self, df0, df1):
        df0 = df0.reshape(-1, 1)
        df1 = df1.reshape(-1, 1)

        if self.method == "kde":
            calibrator0 = KernelDensity()
            calibrator1 = KernelDensity()

        elif self.method == "histogram":
            df_min = max(0, min(np.min(df0), np.min(df1)) - 0.1)
            df_max = min(1, max(np.max(df0), np.max(df1)) + 0.1)

            bins = self.bins
            if self.bins == "auto":
                bins = 10 + int(len(df0) ** (1. / 3.))

            calibrator0 = Histogram(bins=bins,
                                    range=[(df_min, df_max)],
                                    interpolation="linear")
            calibrator1 = Histogram(bins=bins,
                                    range=[(df_min, df_max)],
                                    interpolation="linear")

        else:
            calibrator0 = clone(self.method)
            calibrator1 = clone(self.method)

        calibrator0.fit(df0)
        calibrator1.fit(df1)

        return calibrator0, calibrator1

    def fit(self, X, y=None):
        # XXX: add support for sample_weight

        # Check inputs
        X, y = check_X_y(X, y)

        # Convert y
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y).astype(np.float)

        if len(label_encoder.classes_) != 2:
            raise ValueError

        self.classes_ = label_encoder.classes_

        # Fall back to scikit-learn CalibratedClassifierCV
        if self.method in ("isotonic", "sigmoid"):
            base = self.base_estimator

            if isinstance(base, RegressorMixin):
                base = as_classifier(base)

            clf = CCCV(base, method=self.method, cv=self.cv)
            clf.fit(X, y)

            self.classifiers_ = [clf]

        # Or use carl.distributions distributions
        else:
            if self.cv == "prefit" or self.cv == 1:
                if self.cv == 1:
                    clf = clone(self.base_estimator)

                    if isinstance(clf, RegressorMixin):
                        clf = as_classifier(clf)

                    clf.fit(X, y)

                else:
                    clf = self.base_estimator

                df = clf.predict_proba(X)[:, 0]
                self.classifiers_ = [clf]
                self.calibrators_ = [self._fit_calibrators(df[y == 0],
                                                           df[y == 1])]

            else:
                self.classifiers_ = []
                self.calibrators_ = []

                cv = check_cv(self.cv, X=X, y=y, classifier=True)

                for train, calibrate in cv.split(X, y):
                    clf = clone(self.base_estimator)

                    if isinstance(clf, RegressorMixin):
                        clf = as_classifier(clf)

                    clf.fit(X[train], y[train])
                    df = clf.predict_proba(X[calibrate])[:, 0]

                    self.classifiers_.append(clf)
                    self.calibrators_.append(
                        self._fit_calibrators(df[y[calibrate] == 0],
                                              df[y[calibrate] == 1]))

        return self

    def predict(self, X):
        if self.method in ("isotonic", "sigmoid"):
            return self.classifiers_[0].predict(X)

        else:
            return np.where(self.predict_proba(X)[:, 1] >= 0.5,
                            self.classes_[1],
                            self.classes_[0])

    def predict_proba(self, X):
        if self.method in ("isotonic", "sigmoid"):
            return self.classifiers_[0].predict_proba(X)

        else:
            X = check_array(X)
            p = np.zeros((len(X), 2))

            for classifier, (calibrator0,
                             calibrator1) in zip(self.classifiers_,
                                                 self.calibrators_):
                df = classifier.predict_proba(X)[:, 0].reshape(-1, 1)
                p[:, 0] += calibrator0.pdf(df)
                p[:, 1] += calibrator1.pdf(df)

            p /= np.sum(p, axis=1).reshape(-1, 1)

            return p

    def clone(self):
        estimator = clone(self, original=True)

        if self.cv == "prefit":
            estimator.base_estimator = self.base_estimator

        return estimator
