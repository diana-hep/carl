"""
This module implements machine learning algorithms and utilities,
complementary to Scikit-Learn.
"""

# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

from .base import as_classifier
from .base import check_cv
from .calibration import CalibratedClassifierCV
from .parameterize import make_parameterized_classification
from .parameterize import ParameterStacker
from .parameterize import ParameterizedClassifier
from .parameterize import ParameterizedRegressor


__all__ = ("as_classifier",
           "check_cv",
           "CalibratedClassifierCV",
           "make_parameterized_classification",
           "ParameterStacker",
           "ParameterizedClassifier",
           "ParameterizedRegressor",)
