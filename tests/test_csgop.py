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


class TestTrans:

    def test_scale(self):
        csg = Csg()
        csg.scale(1)
        assert len(csg.trans) == 1
        s = csg.trans[0]
        assert isinstance(s, Scale)
        assert s == Scale(1)
        csg = Csg()
        csg.scale([1,2,3])
        s = csg.trans[0]
        assert isinstance(s, Scale)
        assert s == Scale([1,2,3])
        with pytest.raises(TypeError):
            csg.scale('foo')

    def test_move(self):
        csg = Csg()
        csg.move([1,2,3])
        assert len(csg.trans) == 1
        t = csg.trans[0]
        assert isinstance(t, Translate)
        assert t == Translate([1,2,3])
        with pytest.raises(TypeError):
            csg.move(1)

    def test_rot(self):
        csg = Csg()
        csg.rot(2*np.pi/3, [1,1,1])
        assert len(csg.trans) == 1
        r = csg.trans[0]
        assert isinstance(r, Rotate)
        assert r == Rotate(2*np.pi/3, [1,1,1])
        with pytest.raises(TypeError):
            csg.rot([2,3])

    def test_rot_axis(self):
        csg = Csg()
        csg.rotx(np.pi/4)
        r = csg.trans[0]
        assert r == Rotate(np.pi/4, [1,0,0])
        csg = Csg()
        csg.roty(np.pi/4)
        r = csg.trans[0]
        assert r == Rotate(np.pi/4, [0,1,0])
        csg = Csg()
        csg.rotz(np.pi/4)
        r = csg.trans[0]
        assert r == Rotate(np.pi/4, [0,0,1])
    
    def test_rot_euler(self):
        csg = Csg()
        csg.rotzxz(np.pi/6, np.pi/4, np.pi/3)
        assert len(csg.trans) == 3
        r0, r1, r2 = csg.trans
        assert r0 == Rotate(np.pi/6, [0,0,1])
        assert r1 == Rotate(np.pi/4, [1,0,0])
        assert r2 == Rotate(np.pi/3, [0,0,1])


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

