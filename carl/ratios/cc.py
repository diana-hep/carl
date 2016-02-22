# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from ..distributions import KernelDensity
from ..distributions import Histogram
from ..utils import as_classifier
from ..utils import check_cv
from .base import DensityRatioMixin


class CalibratedClassifierRatio(BaseEstimator, DensityRatioMixin):
    def __init__(self, base_estimator, calibration="histogram", cv=None):
        self.base_estimator = base_estimator
        self.calibration = calibration
        self.cv = cv

    def _fit_X_y(self, X_clf, y_clf, X_cal, y_cal):
        clf = clone(self.base_estimator)

        if isinstance(clf, RegressorMixin):
            clf = as_classifier(clf)

        if self.calibration is None:
            clf.fit(X_clf, y_clf)
            return clf, None, None

        else:
            clf.fit(X_clf, y_clf)
            X_num = clf.predict_proba(X_cal[y_cal == 0])[:, 0]
            X_den = clf.predict_proba(X_cal[y_cal == 1])[:, 0]

            if self.calibration == "kde":
                cal_num = KernelDensity()
                cal_den = KernelDensity()

            elif self.calibration == "histogram":
                eps = 0.05
                X_min = max(0, min(np.min(X_num), np.min(X_den)) - eps)
                X_max = min(1, max(np.max(X_num), np.max(X_den)) + eps)

                cal_num = Histogram(bins=10 + int(len(X_den) ** (1. / 3)),
                                    range=[(X_min, X_max)],
                                    interpolation="linear")
                cal_den = Histogram(bins=10 + int(len(X_den) ** (1. / 3)),
                                    range=[(X_min, X_max)],
                                    interpolation="linear")

            else:
                cal_num = clone(self.calibration)
                cal_den = clone(self.calibration)

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
            X = np.vstack(
                (numerator.rvs(n_samples // 2, **kwargs),
                 denominator.rvs(n_samples - (n_samples // 2), **kwargs)))
            y = np.zeros(n_samples, dtype=np.int)
            y[n_samples // 2:] = 1

        elif X is not None and y is not None:
            pass  # Use given X and y

        else:
            raise ValueError

        self.classifiers_ = []
        self.calibrators_ = []

        if self.calibration in ("isotonic", "sigmoid"):
            # XXX: unify code by using IsotonicRegression instead
            clf = clone(self.base_estimator)

            if isinstance(clf, RegressorMixin):
                clf = as_classifier(clf)

            clf = CalibratedClassifierCV(clf,
                                         method=self.calibration,
                                         cv=self.cv)
            clf.fit(X, y)

            self.classifiers_.append(clf)
            self.calibrators_.append((None, None))

            return self

        else:
            cv = check_cv(self.cv, X=X, y=y, classifier=True)

            for train, calibrate in cv.split(X, y):
                clf, cal_num, cal_den = self._fit_X_y(X[train],
                                                      y[train],
                                                      X[calibrate],
                                                      y[calibrate])
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

                if cal_num is None or cal_den is None:
                    if log:
                        r += np.log(p.ravel()) - np.log(1. - p.ravel())
                    else:
                        r += np.divide(p.ravel(), 1. - p.ravel())

                else:
                    if log:
                        logp_num = -cal_num.nnlf(p, **kwargs)
                        logp_den = -cal_den.nnlf(p, **kwargs)
                        mask = logp_num != logp_den
                        r[mask] += logp_num[mask] - logp_den[mask]

                    else:
                        p_num = cal_num.pdf(p, **kwargs)
                        p_den = cal_den.pdf(p, **kwargs)
                        mask = p_num != p_den
                        r[mask] += p_num[mask] / p_den[mask]
                        r[~mask] += 1.0

            return r / len(self.classifiers_)
