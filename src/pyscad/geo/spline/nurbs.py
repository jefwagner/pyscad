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

