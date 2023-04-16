## @file test_prim.py 
"""\
Validate the creation of CSG primitives
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

from pyscad.csg.prim import *
from pyscad.trans import *

class TestPrimitives:
    """Boo"""

    def test_prim(self):
        p = Prim()
        assert isinstance(p, Csg)
        assert isinstance(p, Prim)
        assert 'pos' in p.params
        assert hasattr(p, 'pos')
        assert np.alltrue( p.pos == np.zeros((3,), dtype=flint) )
        assert len(p.trans) == 1
        t, = p.trans
        assert t == Translate([0,0,0])
        p = Prim(pos=[1,2,3])
        assert np.alltrue( p.pos == np.array([1,2,3], dtype=flint) )
        t, = p.trans
        assert t == Translate([1,2,3])
        with pytest.raises(TypeError):
            Prim([1,2])

    def test_sphere_defaults(self):
        sp = Sphere()
        assert isinstance(sp, Csg)
        assert isinstance(sp, Prim)
        assert isinstance(sp, Sphere)
        assert 'pos' in sp.params
        assert hasattr(sp, 'pos')
        assert np.alltrue( sp.pos == np.zeros((3,), dtype=flint) )
        assert 'r' in sp.params
        assert hasattr(sp, 'r')
        assert sp.r == 1
        assert len(sp.trans) == 2
        sc, t = sp.trans
        assert sc == Scale(1)
        assert t == Translate([0,0,0])

    def test_sphere_params(self):
        # radius 2
        sp = Sphere(2)
        assert np.alltrue( sp.pos == np.zeros((3,), dtype=flint) )
        assert sp.r == 2
        sc, t = sp.trans
        assert sc == Scale(2)
        assert t == Translate([0,0,0])
        # position 1,2,3
        sp = Sphere(pos=[1,2,3])
        assert np.alltrue( sp.pos == np.array([1,2,3], dtype=flint) )
        assert sp.r == 1
        sc, t = sp.trans
        assert sc == Scale(1)
        assert t == Translate([1,2,3])
        # radius and position
        sp = Sphere(2, [1,2,3])
        assert np.alltrue( sp.pos == np.array([1,2,3], dtype=flint) )
        assert sp.r == 2
        sc, t = sp.trans
        assert sc == Scale(2)
        assert t == Translate([1,2,3])
        with pytest.raises(TypeError):
            Sphere('foo')

    def test_box_defaults(self):
        b = Box()
        assert isinstance(b, Csg)
        assert isinstance(b, Prim)
        assert isinstance(b, Box)
        assert 'pos' in b.params
        assert hasattr(b, 'pos')
        assert np.alltrue( b.pos == np.zeros((3,), dtype=flint) )
        assert 'size' in b.params
        assert hasattr(b, 'size')
        assert np.alltrue( b.size == np.ones((3,), dtype=flint) )
        assert len(b.trans) == 2
        sc, t = b.trans
        assert sc == Scale(1)
        assert t == Translate([0,0,0])

    def test_box_params(self):
        # size 2
        b = Box(2)
        assert np.alltrue( b.size == np.array([2,2,2], dtype=flint) )
        sc, t = b.trans
        assert sc == Scale(2)
        assert t == Translate([0,0,0])
        # size [1,2,3]
        b = Box([1,2,3])
        assert np.alltrue( b.size == np.array([1,2,3], dtype=flint) )
        sc, t = b.trans
        assert sc == Scale([1,2,3])
        assert t == Translate([0,0,0])
        with pytest.raises(TypeError):
            Box([1,2])

    def test_cyl_defaults(self):
        cy = Cyl()
        assert isinstance(cy, Csg)
        assert isinstance(cy, Prim)
        assert isinstance(cy, Cyl)
        assert 'pos' in cy.params
        assert hasattr(cy, 'pos')
        assert np.alltrue( cy.pos == np.zeros((3,), dtype=flint) )
        assert 'h' in cy.params
        assert hasattr(cy, 'h')
        assert cy.h == 1
        assert 'r' in cy.params
        assert hasattr(cy, 'r')
        assert cy.r == 1
        assert len(cy.trans) == 2
        sc, t = cy.trans
        assert sc == Scale([1,1,1])
        assert t == Translate([0,0,0])

    def test_cyl_params(self):
        # h 3, r 2
        cy = Cyl(h=3, r=2)
        assert cy.h == 3
        assert cy.r == 2
        sc, t = cy.trans
        assert sc == Scale([2,2,3])
        assert t == Translate([0,0,0])
        with pytest.raises(TypeError):
            cy = Cyl([2,3])

    def test_cone_defaults(self):
        cn = Cone()
        assert isinstance(cn, Csg)
        assert isinstance(cn, Prim)
        assert isinstance(cn, Cone)
        assert 'pos' in cn.params
        assert hasattr(cn, 'pos')
        assert np.alltrue( cn.pos == np.zeros((3,), dtype=flint) )
        assert 'h' in cn.params
        assert hasattr(cn, 'h')
        assert cn.h == 1
        assert 'r' in cn.params
        assert hasattr(cn, 'r')
        assert cn.r == 1
        assert len(cn.trans) == 2
        sc, t = cn.trans
        assert sc == Scale([1,1,1])
        assert t == Translate([0,0,0])

    def tst_cone_params(self):
        # h 3, r 2
        cn = Cone(h=3, r=2)
        assert cn.h == 3
        assert cn.r == 2
        sc, t = cn.trans
        assert sc == Scale([2,2,3])
        assert t == Translate([0,0,0])
        with pytest.raises(TypeError):
            cn = Cone('foo')
