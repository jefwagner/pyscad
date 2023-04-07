import pytest

import numpy as np
import flint as flint
from pyscad.csg.geo import *

class TestGeoTypes:

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


class TestTransform:

    def test_init(self):
        t = Transform()
        assert isinstance(t, Transform)
        assert isinstance(t.m, np.ndarray)
        assert isinstance(t.v, np.ndarray)
        t = Transform(2)
        assert isinstance(t, Transform)
        assert isinstance(t.m, np.ndarray)
        assert isinstance(t.v, np.ndarray)
        with pytest.raises(ValueError):
            t = Transform(4)

    def test_identity(self):
        t = Transform()
        assert np.alltrue(np.eye(3) == t.m)
        assert np.alltrue(np.zeros(3) == t.v)

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
        with pytest.raises(ValueError):
            t == np.eye(3)
        t2 = Transform(2)
        assert t != t2
        t2 = Transform()
        assert t is not t2
        assert t == t2

#     def test_apply_transform(self):
#         assert False

class TestScale:

    def test_init(self):
        s = Scale(2)
        assert isinstance(s, Transform)
        assert isinstance(s, Scale)
        assert len(s) == 3
        assert np.alltrue( s.m == np.array([[2,0,0],[0,2,0],[0,0,2]]) )
        assert np.alltrue( s.v == np.zeros(3) )
