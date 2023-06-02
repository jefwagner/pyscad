## @file test_curve.py 
"""\
Validate test for parametric curve base class.
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

from pyscad.types import *
from pyscad.geo.curve import ParaCurve, Line

class MissingMethods(ParaCurve):
    """Only here to validate errors"""
    pass

class Parabola(ParaCurve):
    """Give a parabola in 2-D"""
    shape = (2,)

    def d_nv(self, x, n):
        """Evaluate the nth order derivatives of the parabola"""
        res = np.zeros((2,), dtype=flint)
        if n == 0:
            res = np.array([x, x*x], dtype=flint)
        elif n == 1:
            res = np.array([1, 2*x], dtype=flint)
        elif n == 2:
            res = np.array([0, 2], dtype=flint)
        return res


class TestParaCurve:
    """Validate the behavior or something"""

    def test_exception(self):
        mm = MissingMethods()
        with pytest.raises(AttributeError):
            mm(1.0)
        with pytest.raises(AttributeError):
            mm.d(1.0)
        with pytest.raises(AttributeError):
            mm.d(1.0,2)

    def test_tangent_scalar(self):
        p = Parabola()
        t = p.t(0.5)
        assert isinstance(t, np.ndarray)
        assert t.shape == (2,)
        assert t.dtype == flint
        assert mag(t) == 1
        assert t[0] == 1/np.sqrt(2)
        assert t[1] == 1/np.sqrt(2)

    def test_tangent_array(self):
        p = Parabola()
        x = [[-0.5,0],[0.5,1]]
        t = p.t(x)
        assert isinstance(t, np.ndarray)
        assert t.shape == (2,2,2)
        assert t.dtype == flint
        roothalf = 1.0/np.sqrt(flint(2))
        tt = t[0,0]
        assert mag(tt) == 1
        assert tt[0] == roothalf
        assert tt[1] == -roothalf
        tt = t[0,1]
        assert mag(tt) == 1
        assert tt[0] == 1
        assert tt[1] == 0
        tt = t[1,0]
        assert mag(tt) == 1
        assert tt[0] == roothalf
        assert tt[1] == roothalf
        tt = t[1,1]
        assert mag(tt) == 1
        assert tt[0] == 1.0/np.sqrt(flint(5))
        assert tt[1] == 2.0/np.sqrt(flint(5))

    def test_curvature_scalar(self):
        p = Parabola()
        k = p.kap(0)
        assert k == 2

    def test_curvature_array(self):
        p = Parabola()
        kn, k0, kp = p.kap([-1,0,1])
        assert k0 == 2
        assert kn == kp


class TestLine:

    def test_init(self):
        L = Line([0,0,0],[0,0,1])
        assert isinstance(L, ParaCurve)
        assert isinstance(L, Line)
        assert L.shape == (3,)
        assert isinstance(L.cpts, np.ndarray)

    def test_eval(self):
        L = Line([0,0,0],[1,2,3])
        assert np.alltrue( L(0) == [0,0,0] )
        assert np.alltrue( L(1) == [1,2,3] )
        assert np.alltrue( L(0.5) == [0.5, 1, 1.5] )

    def test_deriv(self):
        L = Line([1,1,1],[1,2,3])
        for t in np.linspace(0,1,10):
            assert np.alltrue( L.d(t) == [0,1,2] )
        for n in range(2,5):
            for t in np.linspace(0,1,10):
                assert np.alltrue( L.d(t, n) == [0,0,0] )

    def test_paracurve_properties(self):
        L = Line([1,1,1], [1,2,3])
        tan = np.array([0,1,2], dtype=flint)/np.sqrt(flint(5))
        for t in np.linspace(0,1,10):
            assert np.alltrue( L.t(t) == tan )
        for t in np.linspace(0,1,10):
            assert np.alltrue( L.kap(t) == [0,0,0] )