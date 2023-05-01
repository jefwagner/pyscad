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
from pyscad.geo.surf import ParaSurf
from pyscad.geo.spline.kvec import KnotVector
from pyscad.geo.spline.bspline import BSplineCurve, BSplineSurf

def simple_basis(t:float) -> float:
    """Simple basis function of degree 2 for knot-vector (0,1,2,3)"""
    if t < 1:
        return 0.5*t*t
    elif t < 2:
        return 0.5*(-2*t*t+6*t-3)
    else:
        return 0.5*(3-t)*(3-t)

def simple_basis_d1(t:float) -> float:
    """first derivative of the basis function of degree 2 for knot-vector (0,1,2,3)"""
    if t < 1:
        return 1.0*t
    elif t < 2:
        return -2.0*t+3
    else:
        return t-3.0

def simple_basis_d2(t:float) -> float:
    """second derivative of the basis function of degree 2 for knot-vector (0,1,2,3)"""
    if t < 1:
        return 1.0
    elif t < 2:
        return -2.0
    else:
        return 1.0

def cubic_basis(t:float) -> float:
    """Basis function of degree 3 for knot-vector (0,1,2,3,4)"""
    if t < 0.0:
        return 0.0
    elif t < 1.0:
        return t*t*t/6.0
    elif t < 2.0:
        return (-3.0*t*t*t + 12.0*t*t - 12.0*t + 4.0)/6.0
    elif t < 3.0:
        return (3.0*t*t*t - 24.0*t*t + 60.0*t - 44.0)/6.0
    elif x < 4.0:
        return (4.0-t)*(4.0-t)*(4.0-t)/6.0
    else:
        return 0.0

def cubic_basis_d1(t:float) -> float:
    """First derivative for degree 3 spline for knot-vector (0,1,2,3,4)"""
    if t < 0.0:
        return 0.0
    elif t < 1.0:
        return t*t/2.0
    elif t < 2.0:
        return (-9.0*t*t + 24.0*t - 12.0)/6.0
    elif t < 3.0:
        return (9.0*t*t - 48.0*t + 60.0)/6.0
    elif x < 4.0:
        return -(4.0-t)*(4.0-t)/2.0
    else:
        return 0.0

def cubic_basis_d2(t:float) -> float:
    """Second derivative for degree 3 spline for knot-vector (0,1,2,3,4)"""
    if t < 0.0:
        return 0.0
    elif t < 1.0:
        return t
    elif t < 2.0:
        return (-18.0*t + 24.0)/6.0
    elif t < 3.0:
        return (18.0*t - 48.0)/6.0
    elif x < 4.0:
        return (4.0-t)
    else:
        return 0.0

def cubic_basis_d3(t:float) -> float:
    """Third derivative for degree 3 spline for knot-vector (0,1,2,3,4)"""
    if t < 0.0:
        return 0.0
    elif t < 1.0:
        return 1.0
    elif t < 2.0:
        return -3.0
    elif t < 3.0:
        return 3.0
    elif x < 4.0:
        return -1.0
    else:
        return 0.0


class TestBSplineCurveInternal:
    """Validate basis spline internal methods"""

    def test_init(self):
        bs = BSplineCurve([1], 2, [0,1,2,3])
        assert isinstance(bs, ParaCurve)
        assert isinstance(bs, BSplineCurve)
        assert isinstance(bs.cpts, np.ndarray)
        assert bs.cpts.dtype == flint
        assert isinstance(bs.cpts_array, list)
        assert len(bs.cpts_array) == 3
        assert bs.p == 2
        assert isinstance(bs.t, KnotVector)

    def test_exceptions(self):
        with pytest.raises(ValueError):
            bs = BSplineCurve([1],2,[1,2,3])

    def test_deboor(self):
        c = np.array([1], dtype=flint)
        bs = BSplineCurve(c, 2, [0,1,2,3])
        for x in np.linspace(0,3,41):
            assert bs._deboor_1d(c, x) == simple_basis(x)

    def test_calc_d_cpts(self):
        bs = BSplineCurve([1], 2, [0,1,2,3])
        assert bs.cpts_array[1] is None
        assert bs.cpts_array[2] is None
        bs._calc_d_cpts(bs.cpts_array, 2)
        assert np.alltrue( bs.cpts_array[1] == np.array([1,-1]) )
        assert np.alltrue( bs.cpts_array[2] == np.array([1,-2,1]) )

    def test_calc_d_cpts_exceptions(self):
        bs = BSplineCurve([1], 2, [0,1,2,3])
        with pytest.raises(ValueError):
            bs._calc_d_cpts(bs.cpts_array, 0)
        with pytest.raises(ValueError):
            bs._calc_d_cpts(bs.cpts_array, 3)


class TestBSplineCurveEval:
    """Validate the evaluation of the b-spline curve"""

    def test_scalar_spline_call_scalar(self):
        bs = BSplineCurve([1], 2, [0,1,2,3])
        for x in np.linspace(0,3,41):
            assert bs(x) == simple_basis(x)
        
    def test_scalar_spline_call_array(self):
        bs = BSplineCurve([1], 2, [0,1,2,3])
        t = np.linspace(0,3,40).reshape((4,10))
        res = bs(t)
        with np.nditer(t, flags=['multi_index']) as it:
            for x in it:
                assert res[it.multi_index] == simple_basis(x)

    def test_point_spline_call_scalar(self):
        bs = BSplineCurve([[1,-1,0.1]], 2, [0,1,2,3])
        for x in np.linspace(0,3,41):
            sb = simple_basis(x)
            assert np.alltrue( bs(x) == sb*np.array([1,-1,0.1]))
 
    def test_point_spline_call_array(self):
        bs = BSplineCurve([[1,-1,0.1]], 2, [0,1,2,3])
        t = np.linspace(0,3,40).reshape((4,10))
        res = bs(t)
        with np.nditer(t, flags=['multi_index']) as it:
            for x in it:
                target = simple_basis(x)*np.array([1.0,-1.0,0.1])
                assert np.alltrue( res[it.multi_index] == target )


class TestBSplineCurveDerivative:
    """Validate the evaluation of the b-spline curve"""

    def test_scalar_spline_d1_scalar(self):
        bs = BSplineCurve([1], 2, [0,1,2,3])
        for x in np.linspace(0,3,41):
            assert bs.d(x) == simple_basis_d1(x)

    def test_scalar_spline_d2_scalar(self):
        bs = BSplineCurve([1], 2, [0,1,2,3])
        for x in np.linspace(0,3,41):
            assert bs.d(x,2) == simple_basis_d2(x)

    def test_scalar_spline_d1_array(self):
        bs = BSplineCurve([1], 2, [0,1,2,3])
        t = np.linspace(0,3,40).reshape((4,10))
        res = bs.d(t)
        with np.nditer(t, flags=['multi_index']) as it:
            for x in it:
                assert res[it.multi_index] == simple_basis_d1(x)

    def test_scalar_spline_d2_array(self):
        bs = BSplineCurve([1], 2, [0,1,2,3])
        t = np.linspace(0,3,40).reshape((4,10))
        res = bs.d(t, 2)
        with np.nditer(t, flags=['multi_index']) as it:
            for x in it:
                assert res[it.multi_index] == simple_basis_d2(x)

    def test_point_spline_d1_scalar(self):
        bs = BSplineCurve([[1,-1,0.1]], 2, [0,1,2,3])
        for x in np.linspace(0,3,41):
            sbd = simple_basis_d1(x)
            assert np.alltrue( bs.d(x) == sbd*np.array([1,-1,0.1]))

    def test_point_spline_d2_scalar(self):
        bs = BSplineCurve([[1,-1,0.1]], 2, [0,1,2,3])
        for x in np.linspace(0,3,41):
            sbd = simple_basis_d2(x)
            assert np.alltrue( bs.d(x,2) == sbd*np.array([1,-1,0.1]))

    def test_point_spline_d1_array(self):
        bs = BSplineCurve([[1,-1,0.1]], 2, [0,1,2,3])
        t = np.linspace(0,3,40).reshape((4,10))
        res = bs.d(t)
        with np.nditer(t, flags=['multi_index']) as it:
            for x in it:
                sbd = simple_basis_d1(x)
                assert np.alltrue( res[it.multi_index] == sbd*np.array([1,-1,0.1]) )

    def test_point_spline_d2_array(self):
        bs = BSplineCurve([[1,-1,0.1]], 2, [0,1,2,3])
        t = np.linspace(0,3,40).reshape((4,10))
        res = bs.d(t,2)
        with np.nditer(t, flags=['multi_index']) as it:
            for x in it:
                sbd = simple_basis_d2(x)
                assert np.alltrue( res[it.multi_index] == sbd*np.array([1,-1,0.1]) )


class TestBSplineSurfInternal:

    def test_init(self):
        bs = BSplineSurf([[1]], 2, 3, [0,1,2,3], [0,1,2,3,4])
        assert isinstance(bs, ParaSurf)
        assert isinstance(bs, BSplineSurf)
        assert isinstance(bs.cpts, np.ndarray)
        assert bs.cpts.dtype == flint
        assert bs.pu == 2
        assert bs.pv == 3
        assert isinstance(bs.tu, KnotVector)
        assert isinstance(bs.tv, KnotVector)
        assert isinstance(bs.cpts_array, list)
        assert len(bs.cpts_array) == bs.pu+1
        for row in bs.cpts_array:
            assert isinstance(row, list)
            assert len(row) == bs.pv+1

    def test_exception(self):
        with pytest.raises(ValueError):
            BSplineSurf([[1,2]], 2, 3, [0,1,2,3], [0,1,2,3,4])
        with pytest.raises(ValueError):
            BSplineSurf([[1],[2]], 2, 3, [0,1,2,3], [0,1,2,3,4])

    def test_calc_d_cpts_2d(self):
        bs = BSplineSurf([[1]], 2, 3, [0,1,2,3], [0,1,2,3,4])
        assert bs.cpts_array[1][0] is None
        bs._calc_d_cpts_2d( bs.cpts_array, 1, 0)
        assert np.alltrue( bs.cpts_array[1][0] == np.array([[1],[-1]], dtype=flint) )
        assert bs.cpts_array[0][1] is None
        bs._calc_d_cpts_2d( bs.cpts_array, 0, 1)
        assert np.alltrue( bs.cpts_array[0][1] == np.array([[1,-1]]) )

    def test_calc_d_cpts_all(self):
        bs = BSplineSurf([[1]], 2, 3, [0,1,2,3], [0,1,2,3,4])
        bs._calc_d_cpts_2d( bs.cpts_array, 2, 3)
        assert np.alltrue( bs.cpts_array[0][1] == np.array([[1,-1]]) )
        assert np.alltrue( bs.cpts_array[0][2] == np.array([[1,-2,1]]) )
        assert np.alltrue( bs.cpts_array[0][3] == np.array([[1,-3,3,-1]]) )
        target = np.array([[1,-3,3,-1],[-1,3,-3,1]])
        assert np.alltrue( bs.cpts_array[1][3] == target)
        target = np.array([[1,-3,3,-1],[-2,6,-6,2],[1,-3,3,-1]])
        assert np.alltrue( bs.cpts_array[2][3] == target)
        bs._calc_d_cpts_2d( bs.cpts_array, 2, 2)
        target = np.array([[1,-2,1],[-1,2,-1]])
        assert np.alltrue( bs.cpts_array[1][2] == target)
        target = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]])
        assert np.alltrue( bs.cpts_array[2][2] == target)
        bs._calc_d_cpts_2d( bs.cpts_array, 2, 1)
        target = np.array([[1,-1],[-1,1]])
        assert np.alltrue( bs.cpts_array[1][1] == target)
        target = np.array([[1,-1],[-2,2],[1,-1]])
        assert np.alltrue( bs.cpts_array[2][1] == target)
        bs._calc_d_cpts_2d( bs.cpts_array, 2, 0)
        target = np.array([[1],[-1]])
        assert np.alltrue( bs.cpts_array[1][0] == target)
        target = np.array([[1],[-2],[1]])
        assert np.alltrue( bs.cpts_array[2][0] == target)
