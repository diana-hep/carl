# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.


class DensityRatioMixin:
    def fit(self, X=None, y=None, numerator=None,
            denominator=None, n_samples=None, **kwargs):
        return self

    def predict(self, X, log=False, **kwargs):
        raise NotImplementedError

    def score(self, X, y, **kwargs):
        raise NotImplementedError
