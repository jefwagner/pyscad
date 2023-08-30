#!/usr/bin/env python
# -*- coding: utf-8 -*-
## @file setup.py Programmers Solid Modeling in Python
"""\
Standard c compilation setup file for the pyscad python c extension
"""
# Copyright (c) 2023, Jef Wagner <jefwagner@gmail.com>
# This file is part of pyscad.
#
# pyscad is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pyscad is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# pyscad. If not, see <https://www.gnu.org/licenses/>.

from setuptools import setup, Extension
import numpy as np
import flint

setup_args = dict(
    ext_modules = [
        Extension(
            name='pyscad._c_nurbs',
            sources=['src/pyscad/csrc/nurbs.c'],
            depends=[],
            include_dirs=[
                np.get_include(),
                flint.get_include()
            ],
        )
    ]
)

setup(**setup_args)
