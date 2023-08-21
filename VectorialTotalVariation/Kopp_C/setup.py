#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
import numpy


kopp_module = Extension('_kopp',
                           sources=['zheevh3_wrap.c', 'zheevh3.c'],
                           include_dirs=[numpy.get_include()]
                           )

setup (name = 'kopp',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Kopp 3x3 Matrix Diagonalisation kopp""",
       ext_modules = [kopp_module],
       py_modules = ["kopp"],
       )
