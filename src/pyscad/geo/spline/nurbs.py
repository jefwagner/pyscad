## @file nurbs.py 
"""\
NURBS curves and surfaces
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

from typing import Sequence

from ...types import *
from .bspline import BSplineCurve, BSplineSurf

_binom = np.array([
    [1,0,0,0,0],
    [1,1,0,0,0],
    [1,2,1,0,0],
    [1,3,3,1,0],
    [1,4,6,4,1],
])

class NurbsCurve(BSplineCurve):
    """Non-Uniform Rational Basis Spline (NURBS) space curve"""

    def __init__(self, 
                 c: Sequence[Point], 
                 w: Sequence[Num], 
                 p: int, 
                 t: Sequence[Num]):
        """Create a new b-spline object
        @param c The control points
        @param w The control point weights
        @param p Degree of the b-spline basis functions
        @param t The knot-vector
        """
        if len(w) != len(c):
            raise ValueError("Control points and weights be same length")
        super().__init__(c, p, t)
        self.weights = np.array(w, dtype = flint)
        cw = self.cpts*self.weights[...,np.newaxis]
        self.cpts_array[0] = cw
        self.w_array = [self.weights] + [None for _ in range(p)]

    def d(self, x: Num, n: int = 1) -> Point:
        """Evaluate the n^th order derivative of the spline curve
        @param x The parametric point
        @param n The order of the derivative
        @return The value of the derivative curve at the parametric point x
        """
        out_shape = list(np.shape(x)) + list(np.shape(self.cpts[0]))
        out_array = np.zeros(out_shape, dtype=flint)
        with np.nditer(np.array(x), flags=['multi_index']) as it:
            for xx in it:
                c = self.cpts_array[0]
                w = self.w_array[0]
                c_list = [self._deboor_1d(c, xx)]
                w_list = [self._deboor_1d(w, xx)]
                s_list = [c_list[0]/(w_list[0] if w_list[0] != 0 else 1)]
                for i in range(1,n+1):
                    if self.cpts_array[i] is None:
                        self._calc_d_cpts(self.cpts_array, i)
                        self._calc_d_cpts(self.w_array, i)
                    c = self.cpts_array[i]
                    w = self.w_array[i]
                    # wc = c*w[...,np.newaxis]
                    c_list.append(self._deboor_1d(c, xx, i))
                    w_list.append(self._deboor_1d(w, xx, i))
                    res = c_list[-1]
                    for k in range(1,i+1):
                        res -= _binom[i,k]*(s_list[i-k]*w_list[k])
                    s_list.append(res/(w_list[0] if w_list[0] != 0 else 1))
                out_array[it.multi_index] = s_list[-1]
        return out_array


class NurbsSurf(BSplineSurf):
    """Non-uniform rational b-spline surface"""

    def __init__(self,
                 c: Sequence[Sequence[Point]],
                 w: Sequence[Sequence[Num]],
                 pu: int,
                 pv: int,
                 tu: KnotVector,
                 tv: KnotVector):
        """Create a new nurbs surface object
        @param c The 2-D array of control points
        @param pu The degree of the b-spline u function
        @param pv The degree of the b-spline v function
        @param tu The u direction knot-vector
        @param tv The v direction knot-vector
        """
        if len(w) != len(c):
            raise ValueError("Control points and weights be same length")
        super().__init__(c, pu, pv, tu, tv)
        self.weights = np.array(w, dtype = flint)
        if self.cpts.shape[:2] != self.weights.shape[:]:
            raise ValueError("Control point and weight arrays must have the same shape")
        cw = self.cpts*self.weights[...,np.newaxis]
        self.cpts_array[0][0] = cw
        self.w_array = [[None for _ in range(pu+1)] for _ in range(pv+1)]
        self.w_array[0] = self.weights

    def d(self, u: Num, v: Num, nu: int, nv: int) -> Point:
        """Evaluate the (nu, nv) partial derivative of the surface
        @param u The u parameter
        @param v The v parameter
        @param nu The order of the u partial derivative
        @param nv The order of the v partial derivative
        @return The value of the partial derivative at the point (u,v)
        """
        # Holder for output data
        out_shape = list(np.shape(x)) + list(np.shape(self.cpts[0]))
        out_array = np.zeros(out_shape, dtype=flint)
        # Working space
        w_arr_shape = (nu+1, nv+1)
        c_arr_shape = [nu+1, nv+1] + self.cpts[0,0].shape
        c_arr = np.zeros(c_arr_shape, dtype=flint)
        w_arr = np.zeros(w_arr_shape, dtype=flint)
        s_arr = np.zeros(c_arr_shape, dtype=flint)
        with np.nditer([np.array(u),np.array(v)], flags=['multi_index']) as it:
            for uu, vv in it:
                c = self.cpts_array[0][0]
                w = self.w_array[0][0]
                c_arr[0,0] = self._deboor_2d(c, uu, vv)
                _w = self._deboor_2d(w, uu, vv)
                w_arr[0,0] = _w if _w != 0 else flint(1.0)
                s_arr[0,0] = c_arr[0,0]/w[0,0]
                for j in range(1, nv+1):
                    if self.cpts_array[0][j] is None:
                        self._calc_d_cpts_2d(self.cpts_array, 0, j)
                        self._calc_d_cpts_2d(self.w_array, 0, j)
                    c = self.cpts_array[0][j]
                    w = self.w_array[0][j]
                    c_arr[0,j] = self._deboor_2d(c, uu, vv, 0, j)
                    w_arr[0,j] = self._deboor_2d(w, uu, vv, 0, j)
                    s_arr[0,j] = c_arr[0,j]
                    for jj in range(1, j+1):
                        s_arr[0,j] -= _binom[j, jj]*s_arr[0,j-jj]*w_arr[0,jj]
                    s_arr[0,j] /= w_arr[0,j]
                for i in range(1, nu-1):
                    for j in range(nv):
                        if self.cpts_array[i][j] is None:
                            self._calc_d_cpts_2d(self.cpts_array, uu, vv, i, j)
                            self._calc_d_cpts_2d(self.w_array, uu, vv, i, j)
                        c = self.cpts_array[i][j]
                        w = self.w_array[i][j]
                        c_arr[i,j] = self._deboor_2d(c, uu, vv, i, j)
                        w_arr[i,j] = self._deboor_2d(w, uu, vv, i, j)
                        s_arr[i,j] = c_arr[i,j]
                        for ii in range(1, i+1):
                            for jj in range(j+1):
                                term = _binom[i, ii]*_binom[j,jj]
                                term *= s_arr[i-ii,j-jj]*w_arr[ii,jj]
                                s_arr[i,j] -= term
                        s_arr[i,j] /= w_arr[0,0]
                out_array[it.multi_index] = s[nu,nv]
        return out_array
