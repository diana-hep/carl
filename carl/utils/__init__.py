# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Utils."""

import theano
import theano.tensor as T
from theano import Constant
from theano import Variable
from theano.gof import graph
from theano.tensor.sharedvar import SharedVariable


def check_parameter(name, value):
    # Accept expressions only if they depend on bound variables
    if isinstance(value, Variable):
        inputs = graph.inputs([value])
        for var in inputs:
            if (not isinstance(var, SharedVariable) and
                not isinstance(var, Constant)):
                raise ValueError("Variable {} is free, or depends on a "
                                 "free variable.".format(value))

    # Accept constants
    elif isinstance(value, Constant):
        pass

    # Accept raw values, and store them in a shared variable
    else:
        value = theano.shared(value, name=name)

    return value
