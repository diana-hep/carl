# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from sklearn.utils import check_random_state

from carl.data import GK


def test_gk():
    gk = GK(A=3, B=1.5, g=2, k=0.5)
    X = gk.rvs(100)

    assert gk.ndim == 1
    assert X.shape == (100, 1)

    # XXX not sure what else to test?
