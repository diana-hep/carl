# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Distributions.

Note: This module is only meant to be a minimally working prototype. It
      is not meant to be a full fledged replacement of RooFit or alikes.
"""

from .base import DistributionMixin
from .normal import Normal
from .uniform import Uniform
from .mixture import Mixture

__all__ = ("DistributionMixin", "Normal", "Uniform", "Mixture")
