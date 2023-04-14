## @file test_trans.py 
"""\
Validate behavior of Transform type and subtypes
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
import flint as flint
from pyscad.trans import *


class TestTransform:
    """Validate the behavior of Transform types and subtype"""

    def test_init(self):
        t = Transform()
        assert isinstance(t, Transform)
        assert isinstance(t.m, np.ndarray)
        assert isinstance(t.v, np.ndarray)
        assert np.alltrue(np.eye(3) == t.m)
        assert np.alltrue(np.zeros(3) == t.v)
        t = Transform(2)
        assert isinstance(t, Transform)
        assert isinstance(t.m, np.ndarray)
        assert isinstance(t.v, np.ndarray)
        assert np.alltrue(np.eye(2) == t.m)
        assert np.alltrue(np.zeros(2) == t.v)
        with pytest.raises(ValueError):
            t = Transform(4)

    def test_from_array(self):
        m = np.eye(3, dtype=flint)
        v = np.zeros(3, dtype=flint)
        t = Transform.from_arrays(m, v)
        assert isinstance(t, Transform)
        assert t.m is not m
        assert np.alltrue( t.m == m )
        assert t.v is not v
        assert np.alltrue( t.v == v )
        with pytest.raises(TypeError):
            t = Transform([[1,0],[0,1]], [0,0])
        m = np.array([[1,2,3],[4,5,6]])
        v = np.array([1,2])
        with pytest.raises(TypeError):
            t = Transform.from_arrays(m, v)
        with pytest.raises(ValueError):
            t = Transform.from_arrays(m.astype(flint), v.astype(flint))


class TestProperties:

    def test_len(self):
        t = Transform()
        assert len(t) == 3
        t = Transform(2)
        assert len(t) == 2

    def test_dim(self):
        t = Transform()
        assert t.dim == 3
        t = Transform(2)
        assert t.dim == 2

    def test_apply_vec(self):
        t = Transform()
        t.m = np.array([[2,0,0],[0,3,0],[0,0,4]], dtype=flint)
        ones = np.array([1,1,1])
        assert np.alltrue( t(ones) == np.array([2,3,4]) )
        t.m = np.eye(3, dtype=flint)
        t.v = np.array([1,2,3], dtype=flint)
        assert np.alltrue( t(ones) == np.array([2,3,4]) )
        with pytest.raises(ValueError):
            t(np.array([1,1]))

    def test_eq(self):
        t = Transform()
        with pytest.raises(TypeError):
            t == np.eye(3)
        t2 = Transform(2)
        assert t != t2
        t2 = Transform()
        assert t is not t2
        assert t == t2

    def test_apply_transform(self):
        t = Transform()
        t.m = np.array([[2,0,0],[0,3,0],[0,0,4]], dtype=flint)
        t.v = np.ones(3, dtype=flint)
        ident = Transform()
        t2 = ident(t)
        assert isinstance(t2, Transform)
        assert len(t2) == 3
        assert t2 is not t
        assert t2 == t
        t2 = t(ident)
        assert t2 is not t
        assert t2 == t


class TestSubTypes:

    def test_scale(self):
        s = Scale(2)
        assert isinstance(s, Transform)
        assert isinstance(s, Scale)
        assert len(s) == 3
        assert np.alltrue( s.m == np.array([[2,0,0],[0,2,0],[0,0,2]]) )
        assert np.alltrue( s.v == np.zeros(3) )
        s = Scale([2,3])
        assert isinstance(s, Transform)
        assert isinstance(s, Scale)
        assert len(s) == 2
        assert np.alltrue( s.m == np.array([[2,0],[0,3]]) )
        assert np.alltrue( s.v == np.zeros(2) )
        s = Scale([2,3,4])
        assert isinstance(s, Transform)
        assert isinstance(s, Scale)
        assert len(s) == 3
        assert np.alltrue( s.m == np.array([[2,0,0],[0,3,0],[0,0,4]]) )
        assert np.alltrue( s.v == np.zeros(3) )
        with pytest.raises(TypeError):
            s = Scale([2,3,4,5])
        with pytest.raises(TypeError):
            s = Scale("foo")
    
    def test_translate(self):
        t = Translate([1,2])
        assert isinstance(t, Transform)
        assert isinstance(t, Translate)
        assert len(t) == 2
        assert np.alltrue( t.m == np.eye(2) )
        assert np.alltrue( t.v == np.array([1,2]) )
        t = Translate([1,2,3])
        assert isinstance(t, Transform)
        assert isinstance(t, Translate)
        assert len(t) == 3
        assert np.alltrue( t.m == np.eye(3) )
        assert np.alltrue( t.v == np.array([1,2,3]) )
        with pytest.raises(TypeError):
            t = Translate([2,3,4,5])
        with pytest.raises(TypeError):
            t = Translate("foo")

    def test_rotate(self):
        r = Rotate(np.pi/2)
        assert isinstance(r, Transform)
        assert isinstance(r, Rotate)
        assert len(r) == 2
        assert np.alltrue( r.m == np.array([[0,-1],[1,0]]) )
        assert np.alltrue( r.v == np.zeros(2) )
        r = Rotate(np.pi/2, [0,0,1])
        assert isinstance(r, Transform)
        assert isinstance(r, Rotate)
        assert len(r) == 3
        assert np.alltrue( r.m == np.array([[0,-1,0],[1,0,0],[0,0,1]]) )
        assert np.alltrue( r.v == np.zeros(3) )
        with pytest.raises(TypeError):
            r = Rotate(np.pi/2, [0,0])
        with pytest.raises(TypeError):
            r = Rotate("foo")


class TestVerify:

    def test_scale_verify(self):
        m = np.eye(3, dtype=flint)
        v = np.zeros(3, dtype=flint)
        s = Scale.from_arrays(m, v)
        assert isinstance(s, Scale)
        assert s.verify()
        m = np.ones((3,3), dtype=flint)
        with pytest.raises(ValueError):
            s = Scale.from_arrays(m, v)
        m = np.eye(3, dtype=flint)
        v = np.ones(3, dtype=flint)
        with pytest.raises(ValueError):
            s = Scale.from_arrays(m, v)

    def test_translate_verify(self):
        m = np.eye(3, dtype=flint)
        v = np.ones(3, dtype=flint)
        t = Translate.from_arrays(m, v)
        m = np.ones((3,3), dtype=flint)
        with pytest.raises(ValueError):
            t = Translate.from_arrays(m, v)

    def test_rotate_verify(self):
        c = np.cos(flint(2.5))
        s = np.sin(flint(2.5))
        m = np.array([[c, -s],[s, c]])
        v = np.zeros(2, dtype=flint)
        r = Rotate.from_arrays(m, v)
        assert isinstance(r, Rotate)
        m = np.array([[1,2],[3,4]], dtype=flint)
        with pytest.raises(ValueError):
            r = Rotate.from_arrays(m, v)
        m = np.array([[c, -s],[s, c]])
        v = np.ones(2, dtype=flint)
        with pytest.raises(ValueError):
            r = Rotate.from_arrays(m, v)
