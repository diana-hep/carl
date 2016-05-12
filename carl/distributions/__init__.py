"""
This module implements tools for composing and fitting statistical
distributions.

Note
----
This module is only meant to be a minimally working prototype for
composing and fitting distributions. It is not meant to be a full fledged
replacement of RooFit or alikes.
"""

# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

from .base import DistributionMixin
from .base import TheanoDistribution

from .exponential import Exponential
from .normal import Normal
from .normal import MultivariateNormal
from .uniform import Uniform

from .join import Join
from .mixture import Mixture
from .transforms import LinearTransform

from .histogram import Histogram
from .kde import KernelDensity
from .sampler import Sampler

__all__ = ("DistributionMixin",
           "TheanoDistribution",
           "Exponential",
           "Normal",
           "MultivariateNormal",
           "Uniform",
           "Join",
           "Mixture",
           "LinearTransform",
           "Histogram",
           "KernelDensity",
           "Sampler")
