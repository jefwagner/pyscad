## @file test_serde.py 
"""\
Validate serialization and deserialization functions.
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

from pyscad.serde import *
from pyscad.types import *
from pyscad.trans import *
from pyscad.csg import *


class TestParamsSerialize:
    """Validate the serialization for strings, numbers and arrays"""

    def test_simple(self):
        assert ParamSerde.ser("foo") == "foo"
        assert ParamSerde.ser(2) == 2
        assert ParamSerde.ser(2.5) == 2.5
        assert ParamSerde.ser(flint(2)) == {'a': 2, 'b': 2, 'v': 2}
    
    def test_array(self):
        a = [1,2]
        assert ParamSerde.ser(np.array(a)) == a
        a = [[1,2],[3,4]]
        assert ParamSerde.ser(np.array(a)) == a
        fa = [[{'a': np.nextafter(i,-np.inf), 
                'b': np.nextafter(i,np.inf), 
                'v': i} for i in row] for row in a]
        assert ParamSerde.ser(np.array(a, dtype=flint)) == fa

    def test_exceptions(self):
        with pytest.raises(TypeError):
            ParamSerde.ser([1,2,3])
        with pytest.raises(TypeError):
            ParamSerde.ser({'a':1})
        with pytest.raises(TypeError):
            ParamSerde.ser(np.array(['a','b']))


class TestParamsDeserialize:
    """Validate the deserialization for strings, numbers, and arrays"""

    def test_simple(self):
        assert ParamSerde.deser("foo") == "foo"
        assert ParamSerde.deser(2) == 2
        assert ParamSerde.deser(2.5) == 2.5

    def test_flint(self):
        f1 = flint(1.5)
        sf = {'a': np.nextafter(1.5, -np.inf), 'b': np.nextafter(1.5, np.inf), 'v': 1.5}
        f2 = ParamSerde.deser(sf)
        assert isinstance(f2, flint)
        assert f1.a == f2.a
        assert f1.b == f2.b
        assert f1.v == f2.v

    def test_array(self):
        a = [1,2]
        assert np.alltrue( ParamSerde.deser(a) == np.array(a) )
        a = [[1,2],[3,4]]
        assert np.alltrue( ParamSerde.deser(a) == np.array(a) )
        fa = [[{'a': np.nextafter(i,-np.inf), 
                'b': np.nextafter(i,np.inf), 
                'v': i} for i in row] for row in a]
        assert np.alltrue( ParamSerde.deser(fa) == np.array(a, dtype=flint) )

    def test_flint_exceptions(self):
        with pytest.raises(ValueError):
            ParamSerde.deser({'a': 1, 'b': 2})
        with pytest.raises(ValueError):
            ParamSerde.deser({'a': 1, 'b': 2, 'c': 3})

    def test_array_exceptions(self):
        assert np.alltrue( ParamSerde.deser([1,2.5]) == np.array([1.0, 2.5]) )
        with pytest.raises(ValueError):
            ParamSerde.deser([1, {'a':0, 'b':0, 'c':0}])
        with pytest.raises(ValueError):
            ParamSerde.deser([{'a':0, 'b':0, 'c':0}, 1])
        with pytest.raises(ValueError):
            ParamSerde.deser([[1,2],[3,4,5]])
        with pytest.raises(TypeError):
            ParamSerde.deser((1,2))


class TestTransform:

    def test_transform(self):
        m = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=flint)
        v = np.array([-1,-2,-3], dtype=flint)
        t_in = Transform.from_arrays(m, v)
        t_ser = TransSerde.ser(t_in)
        assert isinstance(t_ser, dict)
        assert '__module__' in t_ser.keys()
        assert t_ser['__class__'] == 'Transform'
        assert 'm' in t_ser.keys()
        assert 'v' in t_ser.keys()
        t_out = TransSerde.deser(t_ser)
        assert isinstance(t_out, Transform)
        assert t_in == t_out        

    def test_scale(self):
        t_in = Scale([2,3,4])
        t_ser = TransSerde.ser(t_in)
        assert isinstance(t_ser, dict)
        assert t_ser['__class__'] == 'Scale'
        t_out = TransSerde.deser(t_ser)
        assert isinstance(t_out, Scale)
        assert t_in == t_out

    def test_translate(self):
        t_in = Translate([2,3,4])
        t_ser = TransSerde.ser(t_in)
        assert isinstance(t_ser, dict)
        assert t_ser['__class__'] == 'Translate'
        t_out = TransSerde.deser(t_ser)
        assert isinstance(t_out,Translate)
        assert t_in == t_out

    def test_rotate(self):
        t_in = Rotate(np.pi/3, [2,3,4])
        t_ser = TransSerde.ser(t_in)
        assert isinstance(t_ser, dict)
        assert t_ser['__class__'] == 'Rotate'
        t_out = TransSerde.deser(t_ser)
        assert isinstance(t_out,Rotate)
        assert t_in == t_out