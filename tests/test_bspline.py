## @file test_bspline.py 
"""\
Validate basis spline behavior.
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

from pyscad.geo.curve import ParaCurve
from pyscad.geo.spline.kvec import KnotVector
from pyscad.geo.spline.bspline import BSplineCurve

def simple_basis(t:float) -> float:
    """Simple basis function of degree 2 for knot-vector (0,1,2,3)"""
    if t < 1:
        return 0.5*t*t
    elif t < 2:
        return 0.5*(-2*t*t+6*t-3)
    else:
        return 0.5*(3-t)*(3-t)

class TestBSpline:
    """Validate basis spline behavior"""

    def test_init(self):
        bs = BSplineCurve([1], 2, [0,1,2,3])
        assert isinstance(bs, ParaCurve)
        assert isinstance(bs, BSplineCurve)
        assert isinstance(bs.c, np.ndarray)
        assert bs.c.dtype == flint
        assert isinstance(bs.t, KnotVector)

    def test_deboor(self):
        c = np.array([1], dtype=flint)
        bs = BSplineCurve(c, 2, [0,1,2,3])
        for x in np.linspace(0,3,41):
            assert bs._deboor_1d(c, x) == simple_basis(x)

    def test_scalespline_call_scalar(self):
        bs = BSplineCurve([1], 2, [0,1,2,3])
        for x in np.linspace(0,3,41):
            assert bs(x) == simple_basis(x)
        
    def test_scalarspline_call_array(self):
        bs = BSplineCurve([1], 2, [0,1,2,3])
        t = np.linspace(0,3,40).reshape((4,10))
        res = bs(t)
        with np.nditer(t, flags=['multi_index']) as it:
            for x in it:
                assert res[it.multi_index] == simple_basis(x)

    def test_pointspline_call_scalar(self):
        bs = BSplineCurve([[1,-1,0.1]], 2, [0,1,2,3])
        for x in np.linspace(0,3,41):
            sb = simple_basis(x)
            assert np.alltrue( bs(x) == sb*np.array([1,-1,0.1]))
 
    def test_pointspline_call_array(self):
        bs = BSplineCurve([[1,-1,0.1]], 2, [0,1,2,3])
        t = np.linspace(0,3,40).reshape((4,10))
        res = bs(t)
        with np.nditer(t, flags=['multi_index']) as it:
            for x in it:
                target = simple_basis(x)*np.array([1.0,-1.0,0.1])
                assert np.alltrue( res[it.multi_index] == target )

