"""
The `carl.learning` module defines functions and classes for
machine learning, and that are complementary to Scikit-Learn.
"""

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
    """Wrap a Scikit-Learn regressor into a binary classifier.

    This function can be used to solve a binary classification problem as a
    regression problem, where output labels {0,1} are treated as real values.
    The wrapped regressor exhibits the classifier API, with the corresponding
    `predict`, `predict_proba` and `score` methods.

    Parameters
    ----------
    regressor : RegressorMixin
        The regressor object.

    Returns
    -------
    clf : ClassifierMixin.
        The wrapped regressor, but with a classifier API.
    """
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
    """Input checker utility for building a cross-validator.

    Parameters
    ----------
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if classifier is True and ``y`` is either
        binary or multiclass, :class:`StratifiedKFold` used. In all other
        cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    y : array-like, optional
        The target variable for supervised learning problems.

    classifier : boolean, optional, default False
        Whether the task is a classification task, in which case
        stratified KFold will be used.

    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.

    Note
    ----
    This method is backported from scikit-learn 0.18.
    """
    if int(sklearn.__version__[2:4]) >= 18:
        from sklearn.model_selection import check_cv as sklearn_check_cv
        return sklearn_check_cv(cv, y=y, classifier=classifier)

    else:
        from sklearn.cross_validation import check_cv as sklearn_check_cv
        return _CVIterableWrapper(sklearn_check_cv(cv, X=X, y=y,
                                                   classifier=classifier))
