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
from .bspline import BSplineCurve

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
        super().__init__(c, p, t)
        self.w = [np.array(w, dtype = flint)] + [None for _ in range(p)]
        if len(w) != len(self.c[0]):
            raise ValueError("Control points and weights be same length")

    def _calc_d_weights(self, n: int):
        """Calculate the control points for the derivative spline
        @param n
        """
        if self.w[n-1] is None:
            self._calc_d_weights(n-1)
        new_shape = list(self.w[n-1].shape)
        new_shape[0] += 1
        r = self.w[n-1].copy()
        r.resize(new_shape)
        p = self.p-(n-1)
        for i in range(new_shape[0]-1,-1,-1):
            dt = self.t[i+p]-self.t[i]
            r_im1 = 0*r[0] if i-1 == -1 else r[i-1]
            r[i] = 0*r[0] if dt == 0 else p*(r[i]-r_im1)/dt
        self.w[n] = r

    def __call__(self, x: Num) -> Point:
        """Evaluate the spline curve at a parametric point
        @param x The parametric point
        @return The value of the spline at the parametric point x
        """
        c = self.c[0]
        w = self.w[0]
        wc = c*w[...,np.newaxis]
        out_shape = list(np.shape(x)) + list(np.shape(c[0]))
        out_array = np.zeros(out_shape, dtype=flint)
        with np.nditer(x, flags=['multi_index']) as it:
            for xx in it:
                cc = self._deboor_1d(wc, xx)
                ww = self._deboor_1d(w, xx)
                out_array[it.multi_index] = (cc if ww==0 else cc/ww)
        return out_array

    def d(self, x: Num, n: int = 1) -> Point:
        """Evaluate the n^th order derivative of the spline curve
        @param x The parametric point
        @param n The order of the derivative
        @return The value of the derivative curve at the parametric point x
        """
        out_shape = list(w_shape) + list(np.shape(c[0]))
        out_array = np.zeros(out_shape, dtype=flint)
        with np.nditer(x, flags['multi_index']) as it:
            for xx in it:
                c = self.c[0]
                w = self.w[0]
                wc = c*w[...,np.newaxis]
                c_list = [self._deboor_1d(c, xs)]
                w_list = [self._deboor_1d(w, xx)]
                s_list = [c_list[0]/(w_list[0] if w_list[0] != 0 else 1)]
                for i in range(1,n):
                    if self.c[i] is None:
                        self._calc_d_cpts(i)
                    if self.w[i] in None:
                        self._calc_d_weights(i)
                    c = self.c[i]
                    w = self.w[i]
                    wc = c*w[...,np.newaxis]
                    c_list.append(self._deboor_1d(wc, xx, i))
                    w_list.append(self._deboor_1d(w, xx, i))
                    res = c_list[-1]
                    for k in range(1,i+2):
                        res -= _binom[i+1,k]*(s_list[i+1-k]*w_list[k])
                        s_list.append(res/(w_list[0] if w_list[0] != 0 else 1))
                out_array[it.multi_index] = s_list[-1]
        return out_array

