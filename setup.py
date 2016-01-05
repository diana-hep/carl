# -*- coding: utf-8 -*-
#
# Carl is free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Setup file."""

import os
import re
import sys

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        try:
            from ConfigParser import ConfigParser
        except ImportError:
            from configparser import ConfigParser
        config = ConfigParser()
        config.read("pytest.ini")
        self.pytest_args = config.get("pytest", "addopts").split(" ")

    def finalize_options(self):
        """Finalize options."""
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


packages = find_packages(exclude=["doc", "examples"])

# Get the version string. Cannot be done with import!
with open(os.path.join("carl", "__init__.py"), "rt") as f:
    _version = re.search(
        '__version__\s*=\s*"(?P<version>.*)"\n',
        f.read()
    ).group("version")


_install_requires = [
    "numpy>=1.10",
    "scipy>=0.16",
    "scikit-learn>=0.17",
    "theano>=0.7",
    "six"
]

_tests_require = [
    "pytest",
    "pytest-cov>=1.8.0",
    "pytest-pep8>=1.0.6",
    "coverage"
]

_parameters = {
    "cmdclass": {"test": PyTest},
    "install_requires": _install_requires,
    "license": "BSD",
    "name": "carl",
    "packages": packages,
    "platforms": "any",
    "tests_require": _tests_require,
    "url": "https://github.com/diana-hep/carl",
    "version": _version,
}

setup(**_parameters)
