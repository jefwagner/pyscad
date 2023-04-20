## @file test_kvec.py 
"""\
Validate knot-vector behavior.
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

from pyscad.geo.kvec import KnotVector, KnotMatrix

class TestKnotVector:
    """Validate knot vector behavior"""

    def test_init(self):
        tv = KnotVector([0,0,0,0.5,1,1,1])
        assert isinstance(tv, KnotVector)
        assert isinstance(tv.t, np.ndarray)
        assert tv.t.dtype == flint
        assert len(tv.t) == 7

    def test_init_exception(self):
        with pytest.raises(ValueError):
            tv = KnotVector([1,0])
