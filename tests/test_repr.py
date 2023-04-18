## @file test_repr.py 
"""\
Validate the REPL representation of the objects
"""
# Copyright (c) 2023, Jef Wagner <jefwagner@gmail.com>
#
# This file is part of pyscad.
#
# pyscad is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pyscad is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pyscad. If not, see <https://www.gnu.org/licenses/>.

import pytest

import numpy as np
from flint import flint

from pyscad.csg import Sphere, Box, Cyl, Cone, IntXn, Diff
from pyscad.csg import Union as Uni
from pyscad.trans import *


class TestPrimitives:
    """Validate REPL representation of CSG primatives"""

    def test_sphere(self):
        s = Sphere()
        assert s.__repr__() == 'Sphere'
    
    def test_box(self):
        b = Box()
        assert b.__repr__() == 'Box'

    def test_cyl(self):
        c = Cyl()
        assert c.__repr__() == 'Cyl'

    def test_cone(self):
        c = Cone()
        assert c.__repr__() == 'Cone'


class TestOps:
    """Validate Simple 1-deep CSG trees"""

    def test_union(self):
        u = Uni(Box(), Cyl())
        assert u.__repr__() == 'Box U Cyl'
    
    def test_union_3(self):
        u = Uni(Box(), Cyl(), Cone())
        assert u.__repr__() == 'Box U Cyl U Cone'

    def test_intxn(self):
        x = IntXn(Box(), Cyl())
        assert x.__repr__() == 'Box X Cyl'

    def test_intxn_3(self):
        x = IntXn(Box(), Cyl(), Cone())
        assert x.__repr__() == 'Box X Cyl X Cone'

    def test_diff(self):
        d = Diff(Box(), Cyl())
        assert d.__repr__() == 'Box \\ Cyl'

    def test_diff_3(self):
        d = Diff(Box(), Cyl(), Cone())
        assert d.__repr__() == 'Box \\ (Cyl U Cone)'
