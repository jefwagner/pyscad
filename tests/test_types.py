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
        assert num_ser(2) == 2
        assert num_ser(flint(2)) == {'a': 2, 'b': 2, 'v': 2}
        with pytest.raises(TypeError):
            num_ser('foo')
        with pytest.raises(TypeError):
            num_ser(np.array([1,2,3]))
    
    def test_array(self):
        with pytest.raises(TypeError):
            array_ser('foo')
        with pytest.raises(TypeError):
            array_ser(2)
        with pytest.raises(TypeError):
            array_ser([1,2])
        a = [1,2]
        assert array_ser(np.array(a)) == a
        a = [[1,2],[3,4]]
        assert array_ser(np.array(a)) == a
        fa = [[{'a': np.nextafter(i,-np.inf), 
                'b': np.nextafter(i,np.inf), 
                'v': i} for i in row] for row in a]
        assert array_ser(np.array(a, dtype=flint)) == fa
