---
title: 'carl: a likelihood-free inference toolbox'
tags:
  - likehood-free inference
  - density ratio estimation
  - Python
authors:
 - name: Gilles Louppe
   orcid: 0000-0002-2082-3106
   affiliation: New York University
 - name: Kyle Cranmer
   orcid: 0000-0002-5769-7094
   affiliation: New York University
 - name: Juan Pavez
   orcid: 0000-0002-7205-0053
   affiliation: Federico Santa María University
date: 4 May 2016
bibliography: paper.bib
---

# Summary

Carl is a toolbox for likelihood-free inference in Python.

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

## Approximating likelihood ratios with calibrated classifiers

Methodological details regarding likelihood-free inference with calibrated
classifiers can be found in the companion paper [@Cranmer:2015-llr].

## Future works

Future development aims at providing further density ratio estimation
algorithms, along with alternative algorithms for the likelihood-free setup,
such as Approximate Bayesian Computation (ABC).

# References
