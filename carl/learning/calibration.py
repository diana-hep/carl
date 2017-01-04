"""This module implements calibration wrappers."""

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
    """Probability calibration.

    With this class, the `base_estimator` is fit on the train set of the
    cross-validation generator and the test set is used for calibration. The
    probabilities for each of the folds are then averaged for prediction.
    """

    def __init__(self, base_estimator, method="histogram", bins="auto",
                 interpolation=None, variable_width=False, cv=1):
        """Constructor.

        Parameters
        ----------
        * `base_estimator` [`ClassifierMixin`]:
            The classifier whose output decision function needs to be
            calibrated to offer more accurate predict_proba outputs. If
            `cv=prefit`, the classifier must have been fit already on data.

        * `method` [string]:
            The method to use for calibration. Supported methods include
            `"histogram"`, `"kde"`, `"isotonic"`, `"interpolated-isotonic"` and
            `"sigmoid"`.

        * `bins` [int, default="auto"]:
            The number of bins, if `method` is `"histogram"`.

        * `interpolation` [string, optional]
            Specifies the kind of interpolation between bins as a string
            (`"linear"`, `"nearest"`, `"zero"`, `"slinear"`, `"quadratic"`,
            `"cubic"`), if `method` is `"histogram"`.

        * `variable_dith_width` [boolean, optional]
            If True use equal probability variable length bins, if
            `method` is `"histogram"`.

        * `cv` [integer, cross-validation generator, iterable or `"prefit"`]:
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.

            If `"prefit"` is passed, it is assumed that base_estimator has been
            fitted already and all data is used for calibration. If `cv=1`,
            the training data is used for both training and calibration.
        """
        self.base_estimator = base_estimator
        self.method = method
        self.bins = bins
        self.interpolation = interpolation
        self.variable_width = variable_width
        self.cv = cv

    def fit(self, X, y, sample_weight=None):
        """Fit the calibrated model.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            Training data.

        * `y` [array-like, shape=(n_samples,)]:
            Target values.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
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
            base_calibrator = HistogramCalibrator(
                bins=self.bins, interpolation=self.interpolation,
                variable_width=self.variable_width)
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

                if sample_weight is None:
                    clf.fit(X, y)
                else:
                    clf.fit(X, y, sample_weight=sample_weight)

            else:
                clf = self.base_estimator

            self.classifiers_ = [clf]

            # Calibrator
            calibrator = clone(base_calibrator)
            T = clf.predict_proba(X)[:, 1]

            if sample_weight is None:
                calibrator.fit(T, y)
            else:
                calibrator.fit(T, y, sample_weight=sample_weight)

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

                if sample_weight is None:
                    clf.fit(X[train], y[train])
                else:
                    clf.fit(X[train], y[train],
                            sample_weight=sample_weight[train])

                self.classifiers_.append(clf)

                # Calibrator
                calibrator = clone(base_calibrator)
                T = clf.predict_proba(X[calibrate])[:, 1]

                if sample_weight is None:
                    calibrator.fit(T, y[calibrate])
                else:
                    calibrator.fit(T, y[calibrate],
                                   sample_weight=sample_weight[calibrate])

                self.calibrators_.append(calibrator)

        return self

    def predict(self, X):
        """Predict the targets for `X`.

        Can be different from the predictions of the uncalibrated classifier.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `y` [array, shape=(n_samples,)]:
            The predicted class.
        """
        return np.where(self.predict_proba(X)[:, 1] >= 0.5,
                        self.classes_[1],
                        self.classes_[0])

    def predict_proba(self, X):
        """Predict the posterior probabilities of classification for `X`.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples, n_features)]:
            The samples.

        Returns
        -------
        * `probas` [array, shape=(n_samples, n_classes)]:
            The predicted probabilities.
        """
        p = np.zeros((len(X), 2))

        for clf, calibrator in zip(self.classifiers_, self.calibrators_):
            p[:, 1] += calibrator.predict(clf.predict_proba(X)[:, 1])

        p[:, 1] /= len(self.classifiers_)
        p[:, 0] = 1. - p[:, 1]

        return p

    def _clone(self):
        estimator = clone(self, original=True)
        if self.cv == "prefit":
            estimator.base_estimator = self.base_estimator

        return estimator


class HistogramCalibrator(BaseEstimator, RegressorMixin):
    """Probability calibration through density estimation with histograms."""

    def __init__(self, bins="auto", range=None, eps=0.1,
                 interpolation=None, variable_width=False):
        """Constructor.

        Parameters
        ----------
        * `bins` [string or integer]:
            The number of bins, or `"auto"` to automatically determine the
            number of bins depending on the number of samples.

        * `range` [(lower, upper), optional]:
            The lower and upper bounds. If `None`, bounds are automatically
            inferred from the data.

        * `eps` [float]:
            The margin to the lower and upper bounds.

        * `interpolation` [string, optional]:
            Specifies the kind of interpolation between bins as a string
            (`"linear"`, `"nearest"`, `"zero"`, `"slinear"`, `"quadratic"`,
            `"cubic"`).

        * `variable_width` [boolean, optional]
            If True use equal probability variable length bins.
        """
        self.bins = bins
        self.range = range
        self.eps = eps
        self.interpolation = interpolation
        self.variable_width = variable_width

    def fit(self, T, y, sample_weight=None):
        """Fit using `T`, `y` as training data.

        Parameters
        ----------
        * `T` [array-like, shape=(n_samples,)]:
            Training data.

        * `y` [array-like, shape=(n_samples,)]:
            Training target.

        * `sample_weight` [array-like, shape=(n_samples,), optional]:
            Weights. If set to `None`, all weights will be set to 1.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
        # Check input
        T = column_or_1d(T)
        t0 = T[y == 0]
        t1 = T[y == 1]

        sw0 = None
        if sample_weight is not None:
            sw0 = sample_weight[y == 0]

        sw1 = None
        if sample_weight is not None:
            sw1 = sample_weight[y == 1]

        bins = self.bins
        if self.bins == "auto":
            bins = 10 + int(len(t0) ** (1. / 3.))

        range = self.range
        if self.range is None:
            t_min = max(0, min(np.min(t0), np.min(t1)) - self.eps)
            t_max = min(1, max(np.max(t0), np.max(t1)) + self.eps)
            range = [(t_min, t_max)]
        # Fit
        self.calibrator0 = Histogram(bins=bins, range=range,
                                     interpolation=self.interpolation,
                                     variable_width=self.variable_width)
        self.calibrator1 = Histogram(bins=bins, range=range,
                                     interpolation=self.interpolation,
                                     variable_width=self.variable_width)

        self.calibrator0.fit(t0.reshape(-1, 1), sample_weight=sw0)
        self.calibrator1.fit(t1.reshape(-1, 1), sample_weight=sw1)

        return self

    def predict(self, T):
        """Calibrate data.

        Parameters
        ----------
        * `T` [array-like, shape=(n_samples,)]:
            Data to calibrate.

        Returns
        -------
        * `Tt` [array, shape=(n_samples,)]:
            Calibrated data.
        """
        T = column_or_1d(T).reshape(-1, 1)
        num = self.calibrator1.pdf(T)
        den = self.calibrator0.pdf(T) + self.calibrator1.pdf(T)

        p = num / den
        p[den == 0] = 0.5

        return p


class KernelDensityCalibrator(BaseEstimator, RegressorMixin):
    """Probability calibration with kernel density estimation."""

    def __init__(self, bandwidth=None):
        """Constructor.

        Parameters
        ----------
        * `bandwidth` [string or float, optional]:
            The method used to calculate the estimator bandwidth.
        """
        self.bandwidth = bandwidth

    def fit(self, T, y, sample_weight=None):
        """Fit using `T`, `y` as training data.

        Parameters
        ----------
        * `T` [array-like, shape=(n_samples,)]:
            Training data.

        * `y` [array-like, shape=(n_samples,)]:
            Training target.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
        # Check input
        T = column_or_1d(T)
        assert sample_weight is None  # not supported by KernelDensity

        # Fit
        t0 = T[y == 0]
        t1 = T[y == 1]

        self.calibrator0 = KernelDensity(bandwidth=self.bandwidth)
        self.calibrator1 = KernelDensity(bandwidth=self.bandwidth)

        self.calibrator0.fit(t0.reshape(-1, 1))
        self.calibrator1.fit(t1.reshape(-1, 1))

        return self

    def predict(self, T):
        """Calibrate data.

        Parameters
        ----------
        * `T` [array-like, shape=(n_samples,)]:
            Data to calibrate.

        Returns
        -------
        * `Tt` [array, shape=(n_samples,)]:
            Calibrated data.
        """
        T = column_or_1d(T).reshape(-1, 1)
        num = self.calibrator1.pdf(T)
        den = self.calibrator0.pdf(T) + self.calibrator1.pdf(T)

        p = num / den
        p[den == 0] = 0.5

        return p


class IsotonicCalibrator(BaseEstimator, RegressorMixin):
    """Probability calibration with isotonic regression.

    Note
    ----
    This class backports and extends `sklearn.isotonic.IsotonicRegression`.
    """

    def __init__(self, y_min=None, y_max=None, increasing=True,
                 interpolation=False):
        """Constructor.

        Parameters
        ----------
        * `y_min` [optional]:
            If not `None`, set the lowest value of the fit to `y_min`.

        * `y_max` [optional]:
            If not `None`, set the highest value of the fit to `y_max`.

        * `increasing` [boolean or string, default=`True`]:
            If boolean, whether or not to fit the isotonic regression with `y`
            increasing or decreasing.
            The string value `"auto"` determines whether `y` should increase or
            decrease based on the Spearman correlation estimate's sign.

        * `interpolation` [boolean, default=`False`]:
            Whether linear interpolation is enabled or not.
        """
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.interpolation = interpolation

    def fit(self, T, y, sample_weight=None):
        """Fit using `T`, `y` as training data.

        Parameters
        ----------
        * `T` [array-like, shape=(n_samples,)]:
            Training data.

        * `y` [array-like, shape=(n_samples,)]:
            Training target.

        * `sample_weight` [array-like, shape=(n_samples,), optional]:
            Weights. If set to None, all weights will be set to 1.

        Returns
        -------
        * `self` [object]:
            `self`.

        Notes
        -----
        `T` is stored for future use, as `predict` needs T to interpolate
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
        """Calibrate data.

        Parameters
        ----------
        * `T` [array-like, shape=(n_samples,)]:
            Data to calibrate.

        Returns
        -------
        * `Tt` [array, shape=(n_samples,)]:
            Calibrated data.
        """
        if self.interpolation:
            T = column_or_1d(T)
            return 0.5 * (self.interp1_(T) + self.interp2_(T))

        else:
            return self.ir_.transform(T)


class SigmoidCalibrator(BaseEstimator, RegressorMixin):
    """Probability calibration with the sigmoid method (Platt 2000).

    Note
    ----
    This class backports `sklearn.calibration._SigmoidCalibration`.
    """

    def fit(self, T, y, sample_weight=None):
        """Fit using `T`, `y` as training data.

        Parameters
        ----------
        * `T` [array-like, shape=(n_samples,)]:
            Training data.

        * `y` [array-like, shape=(n_samples,)]:
            Training target.

        * `sample_weight` [array-like, shape=(n_samples,), optional]:
            Weights. If set to `None`, all weights will be set to 1.

        Returns
        -------
        * `self` [object]:
            `self`.
        """
        # Check input
        T = column_or_1d(T)

        # Fit
        self.calibrator_ = _SigmoidCalibration()
        self.calibrator_.fit(T, y, sample_weight=sample_weight)

        return self

    def predict(self, T):
        """Calibrate data.

        Parameters
        ----------
        * `T` [array-like, shape=(n_samples,)]:
            Data to calibrate.

        Returns
        -------
        * `Tt` [array, shape=(n_samples,)]:
            Calibrated data.
        """
        return self.calibrator_.predict(T)
