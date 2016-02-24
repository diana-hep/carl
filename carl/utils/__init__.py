# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Developer utils."""

import numpy as np
import sklearn


if int(sklearn.__version__[2:4]) < 18:
    from abc import ABCMeta
    from abc import abstractmethod
    from six import with_metaclass

    class BaseCrossValidator(with_metaclass(ABCMeta)):
        # Backport from sklearn.model_selection
        def split(self, X, y=None, labels=None):
            X, y, labels = indexable(X, y, labels)
            indices = np.arange(_num_samples(X))
            for test_index in self._iter_test_masks(X, y, labels):
                train_index = indices[np.logical_not(test_index)]
                test_index = indices[test_index]
                yield train_index, test_index

        def _iter_test_masks(self, X=None, y=None, labels=None):
            for test_index in self._iter_test_indices(X, y, labels):
                test_mask = np.zeros(_num_samples(X), dtype=np.bool)
                test_mask[test_index] = True
                yield test_mask

        def _iter_test_indices(self, X=None, y=None, labels=None):
            raise NotImplementedError

        @abstractmethod
        def get_n_splits(self, X=None, y=None, labels=None):
            pass

        def __repr__(self):
            return _build_repr(self)

    class _CVIterableWrapper(BaseCrossValidator):
        # Backport from sklearn.model_selection
        def __init__(self, cv):
            self.cv = cv

        def get_n_splits(self, X=None, y=None, labels=None):
            return len(self.cv)

        def split(self, X=None, y=None, labels=None):
            for train, test in self.cv:
                yield train, test


def check_cv(cv=3, X=None, y=None, classifier=False):
    if int(sklearn.__version__[2:4]) >= 18:
        from sklearn.model_selection import check_cv as sklearn_check_cv
        return sklearn_check_cv(cv, y=y, classifier=classifier)

    else:
        from sklearn.cross_validation import check_cv as sklearn_check_cv
        return _CVIterableWrapper(sklearn_check_cv(cv, X=X, y=y,
                                                   classifier=classifier))
