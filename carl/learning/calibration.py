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

from scipy.interpolate import interp1d
from sklearn.base import TransformerMixin
from sklearn.isotonic import IsotonicRegression

class CalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, method="histogram", cv=1,
                 bins="auto", eps=0.1):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.bins = bins
        self.eps = 0.1

    def _fit_calibrators(self, df0, df1):
        df0 = df0.reshape(-1, 1)
        df1 = df1.reshape(-1, 1)

        if self.method == "kde":
            calibrator0 = KernelDensity()
            calibrator1 = KernelDensity()

        elif self.method == "histogram":
            df_min = max(0, min(np.min(df0), np.min(df1)) - self.eps)
            df_max = min(1, max(np.max(df0), np.max(df1)) + self.eps)

            bins = self.bins
            if self.bins == "auto":
                bins = 10 + int(len(df0) ** (1. / 3.))

            calibrator0 = Histogram(bins=bins,
                                    range=[(df_min, df_max)],
                                    interpolation="linear")
            calibrator1 = Histogram(bins=bins,
                                    range=[(df_min, df_max)],
                                    interpolation="linear")

        elif self.method == "interpolated-isotonic":
            T = np.concatenate((1. - df0.ravel(), 1. - df1.ravel()))
            calibrator0 = InterpolatedIsotonicRegression(out_of_bounds='clip')
            calibrator0.fit(T, np.concatenate((np.zeros(len(df0)),
                                               np.ones(len(df1)))))
            return calibrator0, None

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

        elif self.method == "interpolated-isotonic":
            p = np.zeros((len(X), 2))

            for clf, (calibrator0, _) in zip(self.classifiers_,
                                             self.calibrators_):
                p[:, 1] += calibrator0.predict(clf.predict_proba(X)[:, 1])

            p[:, 1] /= len(self.classifiers_)
            p[:, 0] = 1. - p[:, 1]

            return p

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


class InterpolatedIsotonicRegression(BaseEstimator, TransformerMixin,
                                     RegressorMixin):
    """Interpolated Isotonic Regression model.

        apply linear interpolation to transform piecewise constant isotonic
        regression model into piecewise linear model
    """

    def __init__(self, y_min=None, y_max=None, increasing=True,
                 out_of_bounds='nan'):
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.
        Parameters
        ----------
        X : array-like, shape=(n_samples,)
            Training data.
        y : array-like, shape=(n_samples,)
            Training target.
        sample_weight : array-like, shape=(n_samples,), optional, default: None
            Weights. If set to None, all weights will be set to 1 (equal
            weights).
        Returns
        -------
        self : object
            Returns an instance of self.
        Notes
        -----
        X is stored for future use, as `transform` needs X to interpolate
        new input data.
        """
        self.iso_ = IsotonicRegression(y_min=self.y_min,
                                       y_max=self.y_max,
                                       increasing=self.increasing,
                                       out_of_bounds=self.out_of_bounds)
        self.iso_.fit(X, y, sample_weight=sample_weight)

        p = self.iso_.transform(X)
        change_mask1 = (p - np.roll(p, 1)) > 0
        change_mask2 = np.roll(change_mask1, -1)
        change_mask1[0] = True
        change_mask1[-1] = True
        change_mask2[0] = True
        change_mask2[-1] = True

        self.iso_interp1_ = interp1d(X[change_mask1],
                                     p[change_mask1],
                                     bounds_error=False,
                                     fill_value=(0., 1.))
        self.iso_interp2_ = interp1d(X[change_mask2],
                                     p[change_mask2],
                                     bounds_error=False,
                                     fill_value=(0., 1.))

        return self

    def transform(self, T):
        """Transform new data by linear interpolation
        Parameters
        ----------
        T : array-like, shape=(n_samples,)
            Data to transform.
        Returns
        -------
        T_ : array, shape=(n_samples,)
            The transformed data
        """
        return 0.5 * (self.iso_interp1_(T) + self.iso_interp2_(T))

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
        return self.transform(T)
