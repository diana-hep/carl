# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Machine learning helpers."""

from .base import as_classifier
from .parameterize import ParameterStacker
from .parameterize import ParameterizedClassifier
from .parameterize import ParameterizedRegressor
from .parameterize import make_parameterized_classification

__all__ = ("as_classifier",
           "ParameterStacker",
           "ParameterizedClassifier",
           "ParameterizedRegressor",
           "make_parameterized_classification")
