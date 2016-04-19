# carl

[![Build Status](https://travis-ci.org/diana-hep/carl.svg)](https://travis-ci.org/diana-hep/carl) [![Coverage Status](https://coveralls.io/repos/diana-hep/carl/badge.svg?branch=master&service=github)](https://coveralls.io/github/diana-hep/carl?branch=master) [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.47798.svg)](http://dx.doi.org/10.5281/zenodo.47798)


Carl is a toolbox for likelihood-free inference in Python.

Supported features currently include:

- Composition and fitting of distributions;
- Likelihood-free inference from classifiers;
- Parameterized supervised learning;
- Calibration tools.

Note: `carl` is still in its early stage of development. Join us if you feel
like contributing!


## Documentation

* [Static documentation](http://diana-hep.org/carl).

* Illustrative examples serving as documentation can also be found under the
  [`examples/`](https://github.com/diana-hep/carl/tree/master/examples)
  directory.

* Extended details regarding likelihood-free inference with calibrated
  classifiers can be found in the companion paper _"Approximating Likelihood
  Ratios with Calibrated Discriminative Classifiers", Kyle Cranmer, Juan Pavez,
  Gilles Louppe._
  [http://arxiv.org/abs/1506.02169](http://arxiv.org/abs/1506.02169)



## Installation

The following dependencies are required:

- Numpy >= 1.11
- Scipy >= 0.17
- Scikit-Learn >= 0.18-dev
- Theano >= 0.8

Once satisfied, `carl` can be installed from source using the following
 commands:

```
git clone https://github.com/diana-hep/carl.git
cd carl
python setup.py install
```


## Citation

```
@misc{carl,
  author       = {Gilles Louppe and Kyle Cranmer and Juan Pavez},
  title        = {carl: a likelihood-free inference toolbox},
  month        = mar,
  year         = 2016,
  doi          = {10.5281/zenodo.47798},
  url          = {http://dx.doi.org/10.5281/zenodo.47798}
}
```
