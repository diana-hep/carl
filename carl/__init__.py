# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Carl."""

import sklearn.base
from sklearn.base import clone as sk_clone

__version__ = "0.0"


def clone(estimator, safe=True, original=False):
    # XXX: This is a monkey patch to allow cloning of
    #      CalibratedClassifierCV(cv="prefit"), while keeping the original
    #      base_estimator. Do not reproduce at home!
    if hasattr(estimator, "clone") and not original:
        return estimator.clone()
    else:
        return sk_clone(estimator, safe=safe)

sklearn.base.clone = clone
