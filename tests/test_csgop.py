## @file test_prim.py 
"""\
Validate the creation of CSG operator objects
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

from pyscad.csg import Csg, IntXn, Diff, Sphere, Box, Cyl, Cone
from pyscad.csg.op import Op
from pyscad.csg import Union as Uni
from pyscad.trans import *


class TestOperator:

    def test_union(self):
        u = Uni(Box(), Cyl())
        assert isinstance(u, Csg)
        assert isinstance(u, Op)
        assert isinstance(u, Uni)
        assert hasattr(u, 'children')
        assert len(u.children) == 2
        bx, cy = u.children
        assert isinstance(bx, Box)
        assert isinstance(cy, Cyl)
    
    def test_union_3(self):
        u = Uni(Box(), Cyl(), Cone())
        assert isinstance(u, Csg)
        assert isinstance(u, Op)
        assert isinstance(u, Uni)
        assert len(u.children) == 3
        bx, cy, co = u.children
        assert isinstance(bx, Box)
        assert isinstance(cy, Cyl)
        assert isinstance(co, Cone)

    def test_union_exceptions(self):
        with pytest.raises(TypeError):
            Uni(Box())
        with pytest.raises(TypeError):
            Uni(Box(), 'foo')
        with pytest.raises(TypeError):
            Uni('foo', Cyl())

    def test_intersection(self):
        x = IntXn(Box(), Cyl())
        assert isinstance(x, Csg)
        assert isinstance(x, Op)
        assert isinstance(x, IntXn)

    def test_difference(self):
        d = Diff(Box(), Cyl())
        assert isinstance(d, Csg)
        assert isinstance(d, Op)
        assert isinstance(d, Diff)

