# Contributing

Carl is an open-source project and contributions of all kinds
are welcome.

Code contributions should follow these guidelines:

* all changes by pull request (PR);
* a PR solves one problem (do not mix problems together in one PR) with the
  minimal set of changes;
* describe why you are proposing the changes you are proposing;
* new code needs to come with a test;
* no merging if travis is red.

These are not hard rules to be enforced, but guidelines.


# Developer Install

To start working on `carl` you need to install the following additional
dependencies:

* pytest >= 2.9.1
* pytest-pep8 >= 1.0.6
* coverage >= 4.0.3
* pytest-cov >= 2.2.1
* nose >= 1.3.7

Install `carl` with `python setup.py develop`.

_Note: If you are using `conda` and run into problems with loading shared
libraries when running the tests, try installing the `nomkl` versions of
`numpy` and `scipy`._
