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

    def test_mag(self):
        assert mag(2) == 2
        assert mag(-2) == 2
        assert mag(np.array([3,4],dtype=flint)) == 5
        pts = np.array([
            [[i+1,j+i] for j in range(5)] for i in range(4)
        ], dtype=flint)
        mags = mag(pts)
        assert mags.shape == pts.shape[:-1]
        with np.nditer(mags, flags=['multi_index']) as it:
            for m in it:
                assert mag(pts[it.multi_index]) == m
