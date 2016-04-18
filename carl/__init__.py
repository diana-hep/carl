"""
Carl is a Likelihood-free inference toolbox for Python.

Supported features include:
- `carl.distributions`: Composition and fitting of distributions;
- `carl.ratios`: Likelihood-free density ratio estimation;
- `carl.learning`: Machine learning tools, including tools for
  parameterized supervised learning and calibration.
"""

# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import sklearn.base
from sklearn.base import clone as sk_clone

__version__ = "0.0"


def _clone(estimator, safe=True, original=False):
    # XXX: This is a monkey patch to allow cloning of
    #      CalibratedClassifierCV(cv="prefit"), while keeping the original
    #      base_estimator. Do not reproduce at home!
    if hasattr(estimator, "_clone") and not original:
        return estimator._clone()
    else:
        return sk_clone(estimator, safe=safe)

sklearn.base.clone = _clone
