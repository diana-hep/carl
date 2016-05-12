"""
This module implements generators for likelihood-free inference benchmarks.
"""

# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

from .gk import GK
from .ricker import Ricker

__all__ = ("GK", "Ricker")
