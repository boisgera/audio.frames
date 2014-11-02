#!/usr/bin/env python
# coding: utf-8

# Python 2.7 Standard Library
import doctest

# This package
import audio.frames

#
# Test Runner
# ------------------------------------------------------------------------------
#

# support for `python setup.py test`:
suite = doctest.DocTestSuite(audio.frames)

if __name__ == "__main__":
    doctest.testmod(audio.frames)
