# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from scipy.interpolate import interp1d

from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn.utils import column_or_1d

from ..distributions import KernelDensity
from ..distributions import Histogram
from .base import as_classifier
from .base import check_cv


class CalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, method="histogram", cv=1):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv

    def fit(self, X, y):
        # XXX: add support for sample_weight

        # Check inputs
        X, y = check_X_y(X, y)

        # Convert y
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y).astype(np.float)

        if len(label_encoder.classes_) != 2:
            raise ValueError

        self.classes_ = label_encoder.classes_

        # Calibrator
        if self.method == "histogram":
            base_calibrator = HistogramCalibrator()
        elif self.method == "kde":
            base_calibrator = KernelDensityCalibrator()
        elif self.method == "isotonic":
            base_calibrator = IsotonicCalibrator()
        elif self.method == "interpolated-isotonic":
            base_calibrator = IsotonicCalibrator(interpolation=True)
        elif self.method == "sigmoid":
            base_calibrator = SigmoidCalibrator()
        else:
            base_calibrator = self.method

        # Fit
        if self.cv == "prefit" or self.cv == 1:
            # Classifier
            if self.cv == 1:
                clf = clone(self.base_estimator)

                if isinstance(clf, RegressorMixin):
                    clf = as_classifier(clf)

                clf.fit(X, y)

            else:
                clf = self.base_estimator

            self.classifiers_ = [clf]

            # Calibrator
            calibrator = clone(base_calibrator)
            T = clf.predict_proba(X)[:, 1]
            calibrator.fit(T, y)
            self.calibrators_ = [calibrator]

        else:
            self.classifiers_ = []
            self.calibrators_ = []

            cv = check_cv(self.cv, X=X, y=y, classifier=True)

            for train, calibrate in cv.split(X, y):
                # Classifier
                clf = clone(self.base_estimator)

                if isinstance(clf, RegressorMixin):
                    clf = as_classifier(clf)

                clf.fit(X[train], y[train])
                self.classifiers_.append(clf)

                # Calibrator
                calibrator = clone(base_calibrator)
                T = clf.predict_proba(X[calibrate])[:, 1]
                calibrator.fit(T, y[calibrate])
                self.calibrators_.append(calibrator)

        return self

    def predict(self, X):
        return np.where(self.predict_proba(X)[:, 1] >= 0.5,
                        self.classes_[1],
                        self.classes_[0])

    def predict_proba(self, X):
        p = np.zeros((len(X), 2))

        for clf, calibrator in zip(self.classifiers_, self.calibrators_):
            p[:, 1] += calibrator.predict(clf.predict_proba(X)[:, 1])

        p[:, 1] /= len(self.classifiers_)
        p[:, 0] = 1. - p[:, 1]

        return p

    def clone(self):
        estimator = clone(self, original=True)

        if self.cv == "prefit":
            estimator.base_estimator = self.base_estimator

        return estimator


class HistogramCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, bins="auto", range=None, interpolation="linear",
                 eps=0.1):
        self.bins = bins
        self.range = range
        self.interpolation = interpolation
        self.eps = eps

    def fit(self, T, y, sample_weight=None):
        # Check input
        T = column_or_1d(T)

        # Fit
        t0 = T[y == 0]
        t1 = T[y == 1]

        bins = self.bins
        if self.bins == "auto":
            bins = 10 + int(len(t0) ** (1. / 3.))

        range = self.range
        if self.range is None:
            t_min = max(0, min(np.min(t0), np.min(t1)) - self.eps)
            t_max = min(1, max(np.max(t0), np.max(t1)) + self.eps)
            range = [(t_min, t_max)]

        self.calibrator0 = Histogram(bins=bins, range=range,
                                     interpolation=self.interpolation)
        self.calibrator1 = Histogram(bins=bins, range=range,
                                     interpolation=self.interpolation)

        self.calibrator0.fit(t0.reshape(-1, 1))
        self.calibrator1.fit(t1.reshape(-1, 1))

        return self

    def predict(self, T):
        T = column_or_1d(T).reshape(-1, 1)
        num = self.calibrator1.pdf(T)
        den = self.calibrator0.pdf(T) + self.calibrator1.pdf(T)

        p = num / den
        p[den == 0] = 0.5

        return p


class KernelDensityCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def fit(self, T, y, sample_weight=None):
        # Check input
        T = column_or_1d(T)

        # Fit
        t0 = T[y == 0]
        t1 = T[y == 1]

        self.calibrator0 = KernelDensity(bandwidth=self.bandwidth)
        self.calibrator1 = KernelDensity(bandwidth=self.bandwidth)

        self.calibrator0.fit(t0.reshape(-1, 1))
        self.calibrator1.fit(t1.reshape(-1, 1))

        return self

    def predict(self, T):
        T = column_or_1d(T).reshape(-1, 1)
        num = self.calibrator1.pdf(T)
        den = self.calibrator0.pdf(T) + self.calibrator1.pdf(T)

        p = num / den
        p[den == 0] = 0.5

        return p


class IsotonicCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, y_min=None, y_max=None, increasing=True,
                 interpolation=False):
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.interpolation = interpolation

    def fit(self, T, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        T : array-like, shape=(n_samples,)
            Training data.

        y : array-like, shape=(n_samples,)
            Training target.

        sample_weight : array-like, shape=(n_samples,), optional
            Weights. If set to None, all weights will be set to 1.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        T is stored for future use, as `transform` needs T to interpolate
        new input data.
        """
        # Check input
        T = column_or_1d(T)

        # Fit isotonic regression
        self.ir_ = IsotonicRegression(y_min=self.y_min,
                                      y_max=self.y_max,
                                      increasing=self.increasing,
                                      out_of_bounds="clip")
        self.ir_.fit(T, y, sample_weight=sample_weight)

        # Interpolators
        if self.interpolation:
            p = self.ir_.transform(T)

            change_mask1 = (p - np.roll(p, 1)) > 0
            change_mask2 = np.roll(change_mask1, -1)
            change_mask1[0] = True
            change_mask1[-1] = True
            change_mask2[0] = True
            change_mask2[-1] = True

            self.interp1_ = interp1d(T[change_mask1], p[change_mask1],
                                     bounds_error=False,
                                     fill_value=(0., 1.))
            self.interp2_ = interp1d(T[change_mask2], p[change_mask2],
                                     bounds_error=False,
                                     fill_value=(0., 1.))

        return self

    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like, shape=(n_samples,)
            Data to transform.

        Returns
        -------
        T_ : array, shape=(n_samples,)
            Transformed data.
        """
        if self.interpolation:
            T = column_or_1d(T)
            return 0.5 * (self.interp1_(T) + self.interp2_(T))

        else:
            return self.ir_.transform(T)


class SigmoidCalibrator(BaseEstimator, RegressorMixin):
    def fit(self, T, y, sample_weight=None):
        # Check input
        T = column_or_1d(T)

        # Fit
        self.calibrator_ = _SigmoidCalibration()
        self.calibrator_.fit(T, y, sample_weight=sample_weight)

        return self

    def predict(self, T):
        return self.calibrator_.predict(T)
