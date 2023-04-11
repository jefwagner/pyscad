## @file test_types.py 
"""\
Validate test for generic number, vector, and point types.
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
from pyscad.types import *

class TestTypes:
    """Validate the test for generic types"""

    def test_num(self):
        assert is_num(2)
        assert is_num(2.5)
        assert is_num(flint(2.5))
        assert not is_num([1,2])
        assert not is_num(np.array([1,2]))
        assert not is_num(np.array([1,2], dtype=flint))
        assert not is_num('foo')

    def test_vec(self):
        assert not is_vec(2)
        assert not is_vec(2.5)
        assert not is_vec(flint(2.5))
        assert is_vec([1,2])
        assert is_vec(np.array([1,2]))
        assert is_vec(np.array([1,2], dtype=flint))
        assert not is_vec([1,2,3,4])
        assert not is_vec('foo')
        assert is_vec([1,2], length=2)
        assert not is_vec([1,2], length=3)

    def test_point(self):
        assert is_point(2)
        assert is_point(2.5)
        assert is_point(flint(2.5))
        assert is_point([1,2])
        assert is_point(np.array([1,2]))
        assert is_point(np.array([1,2], dtype=flint))
        assert not is_point([1,2,3,4])
        assert not is_point('foo')


class TestJson:
    """Validate the serialization for numbers and arrays"""

    def test_num(self):
        assert num_json(2) == 2
        assert num_json(flint(2)) == {'type': 'flint', 'a': 2, 'b': 2, 'v': 2}
        with pytest.raises(TypeError):
            num_json('foo')
        with pytest.raises(TypeError):
            num_json(np.array([1,2,3]))
    
    def test_array(self):
        with pytest.raises(TypeError):
            array_json('foo')
        with pytest.raises(TypeError):
            array_json(2)
        with pytest.raises(TypeError):
            array_json([1,2])
        a = [1,2]
        assert array_json(np.array(a)) == a
        a = [[1,2],[3,4]]
        assert array_json(np.array(a)) == a
        fa = [[{'type': 'flint', 
                'a': np.nextafter(i,-np.inf), 
                'b': np.nextafter(i,np.inf), 
                'v': i} for i in row] for row in a]
        assert array_json(np.array(a, dtype=flint)) == fa


class TestLinAlg:
    """Validate the linear algrebra routines"""

    def test_det2(self):
        assert 1 == det2(np.eye(2, dtype=flint))
        assert 0 == det2(np.array([[1,1],[2,2]], dtype=flint))
        assert -1 == det2(np.array([[0,1],[1,0]], dtype=flint))
        for i in range(20):
            th = flint(np.pi*i/20)
            c = np.cos(th)
            s = np.sin(th)
            assert 1 == det2(np.array([[c,-s],[s,c]]))
        for i in range(20):
            assert i*i == det2(i*np.eye(2, dtype=flint))

    def test_det3(self):
        assert 1 == det3(np.eye(3, dtype=flint))
        a = np.array([
            [1,1,1],
            [2,2,2],
            [1,2,3]
        ], dtype=flint)
        assert 0 == det3(a)
        a = np.array([
            [0,1,0],
            [1,0,0],
            [0,0,1]
        ], dtype=flint)
        assert -1 == det3(a)
        for i in range(20):
            assert i*i*i == det3(i*np.eye(3, dtype=flint))

    def test_det(self):
        assert 1 == det(np.eye(2, dtype=flint))
        assert 1 == det(np.eye(3, dtype=flint))

    def test_eig2_diag(self):
        # All equal
        l, v = eig2(np.eye(2, dtype=flint))
        assert np.alltrue( l == np.ones((2,)) )
        assert np.alltrue( v == np.eye(2) )
        assert np.cross(v[0],v[1]) == 1
        # In order
        l, v = eig2(np.array([[2,0],[0,1]], dtype=flint))
        assert np.alltrue( l == np.array([2,1]) )
        assert np.alltrue( v == np.eye(2) )
        assert np.cross(v[0],v[1]) == 1
        # out of order
        l, v = eig2(np.array([[1,0],[0,2]], dtype=flint))
        assert np.alltrue( l == np.array([2,1]) )
        assert np.alltrue(v == np.array([[0,1],[-1,0]]) )
        assert np.cross(v[0],v[1]) == 1

    def tst_eig2_full(self):
        # Pauli X spin matrix
        l, v = eig2(np.array([[0,1],[1,0]], dtype=flint))
        assert np.alltrue( l == np.array([1,-1]) )
        assert np.alltrue( v*np.sqrt(2) == np.array([[1,1],[-1,1]]) )
        assert np.cross(v[0],v[1]) == 1
        # Full

    def test_eig3_diag(self):
        # All equal
        l, v = eig3(np.eye(3, dtype=flint))
        assert np.alltrue( l == np.ones((3,)) )
        assert np.alltrue( v == np.eye(3) )
        assert np.dot(v[0], np.cross(v[1],v[2])) == 1
        # Testing all orderings
        for ord in ([3,2,1], [2,1,3], [1,3,2], [1,2,3], [2,3,1], [3,1,2]):
            l, v = eig3(np.diag(np.array(ord, dtype=flint)))
            assert np.alltrue( l == np.array([3,2,1]) )
            assert np.dot(v[0], np.cross(v[1],v[2])) == 1
    
    def test_eig3_full(self):
        a = np.array([
            [4,-1,-2],
            [-1,3,-1],
            [-2,-1,4]
        ], dtype=flint)
        l, v = eig3(a)
        assert np.alltrue( l == np.array([6, 4, 1]))
        assert np.dot(v[0], np.cross(v[1],v[2])) == 1
