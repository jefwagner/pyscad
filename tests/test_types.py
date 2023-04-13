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

    # list of 2x2 symmetric matrices that can be used for testing
    # https://ece.uwaterloo.ca/~dwharder/Integer_eigenvalues/Symmetric_2_by_2_invertible_matrices/
    def tst_eig2_full(self):
        # Pauli X spin matrix
        l, v = eig2(np.array([[0,1],[1,0]], dtype=flint))
        assert np.alltrue( l == np.array([1,-1]) )
        assert np.alltrue( v*np.sqrt(2) == np.array([[1,1],[-1,1]]) )
        assert np.cross(v[0],v[1]) == 1
        # Full matrix
        l, v = eig2(np.array([[2,-1],[-1,2]], dtype=flint))
        # test eigenvalues
        assert np.alltrue( l == np.array[3,1] )
        # Test eigenvector property
        for ll, vv in zip(l, v):
            assert np.alltrue( a.dot(vv) == ll*vv )
        # Test v is unitary
        assert np.det(v) == 1
        # Test that the eigenvectors are orthogonal
        assert np.alltrue( v.dot(v.T) == np.eye(2) )

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
            assert det(v) == 1

    # list of 3x3 symmetric matrices that can be used for testing
    # https://ece.uwaterloo.ca/~dwharder/Integer_eigenvalues/Symmetric_3_by_3_invertible_matrices/    
    # Note: The eigenvalue property for eigenvectors with zero components
    # seems to fail the equality test.
    def test_eig3_full(self):
        a = np.array([
            [4,-1,-2],
            [-1,3,-1],
            [-2,-1,4]
        ], dtype=flint)
        l, v = eig3(a)
        lt = np.array([6, 4, 1])
        # confirm eigenvalues match
        assert np.alltrue( l == lt )
        # # Test eigenvalue property
        # for ll, vv in zip(l, v):
        #     assert np.alltrue( a.dot(vv) == ll*vv )
        # Confirm orientation of matrix of eigenvalues is unitary
        assert det(v) == 1
        # Confirm the eigenvectors are orthogonal
        assert np.alltrue( v.dot(v.T) == np.eye(3) )

    def test_eig_full(self):
        a = np.array([[3,-1],[-1,3]], dtype=flint)
        lt = np.array([4,2])
        l, v = eig(a)
        assert np.alltrue( l == lt )
        for ll, vv in zip(l, v):
            assert np.alltrue( a.dot(vv) == ll*vv )
        assert det(v) == 1
        np.alltrue( v.dot(v.T) == np.eye(len(a)) )
        a = np.array([
            [4,3,-3],
            [3,4,-3],
            [-3,-3,4]
        ], dtype=flint)
        lt = np.array([10,1,1])
        l, v = eig(a)
        assert np.alltrue( l == lt )
        # for ll, vv in zip(l, v):
        #     assert np.alltrue( a.dot(vv) == ll*vv )
        assert det(v) == 1
        np.alltrue( v.dot(v.T) == np.eye(len(a)) )

    def test_svd2(self):
        thu = flint(np.pi/4)
        thv = flint(np.pi/3)
        cu = np.cos(thu)
        su = np.sin(thu)
        cv = np.cos(thv)
        sv = np.sin(thv)
        ut = np.array([[cu,-su],[su,cu]])
        vt = np.array([[cv,-sv],[sv,cv]])
        sigt = np.array([[2,0],[0,1]], dtype=flint)
        a = ut.dot(sigt).dot(vt.T)
        u, sig, v = svd(a)
        assert np.alltrue( sig == np.array([2,1]) )
        # assert np.alltrue( ut == u )
        # assert np.alltrue( vt == v.T )

    def test_svd3(self):
        thu = flint(np.pi/4)
        thv = flint(np.pi/3)
        cu = np.cos(thu)
        su = np.sin(thu)
        cv = np.cos(thv)
        sv = np.sin(thv)
        ut = np.array([[cu,-su,0],[su,cu,0],[0,0,1]], dtype=flint)
        vt = np.array([[1,0,0],[0,cv,-sv],[0,sv,cv]], dtype=flint)
        sigt = np.array([[3,0,0],[0,2,0],[0,0,1]], dtype=flint)
        a = ut.dot(sigt).dot(vt.T)
        u, sig, v = svd(a)
        assert np.alltrue( sig == np.array([3,2,1]) )
        for vv in u.T:
            print(np.sqrt(np.sum(vv*vv)))
        # assert np.alltrue( ut == u )
        assert np.alltrue( vt == v.T )

