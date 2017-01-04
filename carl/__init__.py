"""
`carl` is a toolbox for likelihood-free inference in Python.

The likelihood function is the central object that summarizes the information
from an experiment needed for inference of model parameters. It is key to many
areas of science that report the results of classical hypothesis tests or
confidence intervals using the (generalized or profile) likelihood ratio as a
test statistic. At the same time, with the advance of computing technology, it
has become increasingly common that a simulator (or generative model) is used to
describe complex processes that tie parameters of an underlying theory and
measurement apparatus to high-dimensional observations. However, directly
evaluating the likelihood function in these cases is often impossible or is
computationally impractical.

In this context, the goal of this package is to provide tools for the
likelihood-free setup, including likelihood (or density) ratio estimation
algorithms, along with helpers to carry out inference on top of these.

_This project is still in its early stage of development.
[Join us on GitHub](https://github.com/diana-hep/carl) if you feel like
contributing!_

[![Build Status](https://travis-ci.org/diana-hep/carl.svg)](https://travis-ci.org/diana-hep/carl)
[![Coverage Status](https://coveralls.io/repos/diana-hep/carl/badge.svg?branch=master&service=github)](https://coveralls.io/github/diana-hep/carl?branch=master)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.47798.svg)](http://dx.doi.org/10.5281/zenodo.47798)


## Likelihood-free inference with calibrated classifiers

Extensive details regarding likelihood-free inference with calibrated
classifiers can be found in the companion paper _"Approximating Likelihood
Ratios with Calibrated Discriminative Classifiers", Kyle Cranmer, Juan Pavez,
Gilles Louppe._
[http://arxiv.org/abs/1506.02169](http://arxiv.org/abs/1506.02169)


## Installation

The following dependencies are required:

- Numpy >= 1.11
- Scipy >= 0.17
- Scikit-Learn >= 0.18
- Theano >= 0.8
- Astropy >= 1.3

Once satisfied, Carl can be installed from source using the following
commands:

    git clone https://github.com/diana-hep/carl.git
    cd carl
    python setup.py install


## Citation

    @misc{carl,
      author       = {Gilles Louppe and Kyle Cranmer and Juan Pavez},
      title        = {carl: a likelihood-free inference toolbox},
      month        = mar,
      year         = 2016,
      doi          = {10.5281/zenodo.47798},
      url          = {http://dx.doi.org/10.5281/zenodo.47798}
    }

"""

# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

import sklearn.base
from sklearn.base import clone as sk_clone

__version__ = "0.2"


def _clone(estimator, safe=True, original=False):
    # XXX: This is a monkey patch to allow cloning of
    #      CalibratedClassifierCV(cv="prefit"), while keeping the original
    #      base_estimator. Do not reproduce at home!
    if hasattr(estimator, "_clone") and not original:
        return estimator._clone()
    else:
        return sk_clone(estimator, safe=safe)

sklearn.base.clone = _clone
