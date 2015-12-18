# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.


class DensityRatioMixin:
    def fit(self, X=None, y=None, **kwargs):
        return self

    def predict(self, X=None, **kwargs):
        raise NotImplementedError

    def score(self, X=None, y=None, **kwargs):
        raise NotImplementedError
