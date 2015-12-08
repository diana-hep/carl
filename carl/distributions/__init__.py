# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Distributions."""

from .base import DistributionMixin
from .normal import Normal
from .uniform import Uniform

__all__ = ("DistributionMixin", "Normal", "Uniform")
