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
from pyscad.geo.surf import ParaSurf
from pyscad.geo.spline.kvec import KnotVector
from pyscad.geo.spline.bspline import BSplineCurve, BSplineSurf
from pyscad.geo.spline.nurbs import NurbsCurve, NurbsSurf


@pytest.fixture
def quarter_circle():
    return NurbsCurve(
        [[1,0],[1,1],[0,1]],
        [1,1/np.sqrt(2),1],
        2,
        [0,0,0,1,1,1]
    )

class TestNurbsCurve:
    """Validate the nurbs quarter circle curve"""

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

    def test_call_scalar(self, quarter_circle):
        qc = quarter_circle
        for t in np.linspace(0,1,10):
            p = qc(t)
            assert mag(p) == 1

    def test_call_array(self, quarter_circle):
        qc = quarter_circle
        t = np.linspace(0,1,20).reshape((4,5))
        p = qc(t)
        assert p.shape == (4,5,2)
        assert np.alltrue( mag(p) == np.ones((4,5)) )
 
    def test_derivative_scalar(self, quarter_circle):
        qc = quarter_circle
        d = qc.d(0)
        assert d[0] == 0
        d = qc.d(0.5)
        assert d[0] == -d[1]
        d = qc.d(1)
        assert d[1] == 0

    def test_derivative_array(self, quarter_circle):
        qc = quarter_circle
        d0, d1, d2 = qc.d([0,0.5,1])
        assert d0[0] == 0
        assert d1[0] == -d1[1]
        assert d2[1] == 0


@pytest.fixture
def torus():
    cpts = [
        [[4,0,0],[4,0,1],[3,0,1]],
        [[4,4,0],[4,4,1],[3,3,1]],
        [[0,4,0],[0,4,1],[0,3,1]],
    ]
    _root2 = np.sqrt(flint(2))
    weights = [
        [1, 1/_root2, 1],
        [1/_root2, 0.5, 1/_root2],
        [1, 1/_root2, 1]
    ]
    kv = [0,0,0,1,1,1]
    return NurbsSurf(cpts, weights, 2, 2, kv, kv)

class TestNurbsSurf:

    def test_init(self):
        ns = NurbsSurf([[[1,-1,-.1]]], [[1]], 2, 3, [0,1,2,3],[0,1,2,3,4])
        assert isinstance(ns, NurbsSurf)
        assert isinstance(ns, BSplineSurf)
        assert isinstance(ns, ParaSurf)
        assert isinstance(ns.weights, np.ndarray)
        assert ns.weights.dtype == flint
        assert isinstance(ns.w_array, list)
        assert len(ns.w_array) == ns.pu+1
        for row in ns.w_array:
            assert isinstance(row, list)
            assert len(row) == ns.pv+1

    def test_exceptions(self):
        with pytest.raises(ValueError):
            ns = NurbsSurf([[[1,-1,-.1]]], [[1,2]], 2, 3, [0,1,2,3],[0,1,2,3,4])

    def test_call_scalar(self, torus):
        nt = torus
        test_vals = [
            ((0,0), (4,0,0)),
            ((0,1), (3,0,1)),
            ((1,0), (0,4,0)),
            ((1,1), (0,3,1)),
        ]
        for uv, res in test_vals:
            assert np.alltrue( nt(*uv) == np.array(res) )

    def test_call_array(self, torus):
        nt = torus
        U, V = np.meshgrid([0,1],[0,1])
        target = np.array([
            [[4,0,0],[0,4,0]],
            [[3,0,1],[0,3,1]]
        ])
        res = nt(U,V)
        assert np.alltrue( nt(U, V) == target )
