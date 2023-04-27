## @file test_nurbs.py 
"""\
Validate NURBS behavior.
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

from pyscad.types import mag
from pyscad.geo.curve import ParaCurve
from pyscad.geo.spline.kvec import KnotVector
from pyscad.geo.spline.bspline import BSplineCurve
from pyscad.geo.spline.nurbs import NurbsCurve


class TestNurbsCurveInternal:
    """Validate 1-d nurbs curve internal methods"""

    def test_init(self):
        nc = NurbsCurve([1],[1],2,[0,1,2,3])
        assert isinstance(nc, ParaCurve)
        assert isinstance(nc, BSplineCurve)
        assert isinstance(nc, NurbsCurve)
        assert isinstance(nc.weights, np.ndarray)
        assert nc.weights.dtype == flint
        assert isinstance(nc.w_array, list)
        assert len(nc.w_array) == 3

    def test_exceptions(self):
        with pytest.raises(ValueError):
            NurbsCurve([1],[1,2],2,[0,1,2,3])


class TestNurbsQuarterCircle:
    """Validate the nurbs quarter circle curve"""

    def test_call_scalar(self):
        qc = NurbsCurve(
            [[1,0],[1,1],[0,1]],
            [1,1/np.sqrt(2),1],
            2,
            [0,0,0,1,1,1]
        )
        for t in np.linspace(0,1,10):
            p = qc(t)
            assert mag(p) == 1

    def test_call_array(self):
        qc = NurbsCurve(
            [[1,0],[1,1],[0,1]],
            [1,1/np.sqrt(2),1],
            2,
            [0,0,0,1,1,1]
        )
        t = np.linspace(0,1,20).reshape((4,5))
        p = qc(t)
        assert p.shape == (4,5,2)
        assert np.alltrue( mag(p) == np.ones((4,5)) )
 
    def test_derivative_scalar(self):
        qc = NurbsCurve(
            [[1,0],[1,1],[0,1]],
            [1,1/np.sqrt(2),1],
            2,
            [0,0,0,1,1,1]
        )
        d = qc.d(0)
        assert d[0] == 0
        d = qc.d(0.5)
        assert d[0] == -d[1]
        d = qc.d(1)
        assert d[1] == 0

    def test_derivative_array(self):
        qc = NurbsCurve(
            [[1,0],[1,1],[0,1]],
            [1,1/np.sqrt(2),1],
            2,
            [0,0,0,1,1,1]
        )
        d0, d1, d2 = qc.d([0,0.5,1])
        assert d0[0] == 0
        assert d1[0] == -d1[1]
        assert d2[1] == 0
