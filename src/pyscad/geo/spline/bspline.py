## @file bspline.py 
"""\
Basis spline curves and surfaces
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
import numpy.typing as npt

import numpy as np
from flint import flint

from ...types import *
from ..curve import ParaCurve
from .kvec import KnotVector

class BSplineCurve(ParaCurve):
    """Basis Spline"""

    def __init__(self, c: Sequence[Point], p: int, t: Sequence[Num]):
        """Create a new b-spline object
        @param The control points
        @param p Degree of the b-spline basis functions
        @param t The knot-vector
        """
        if len(t) != len(c) + p + 1:
            raise ValueError("Knot vector wrong length for control points")
        self.cpts = np.array(c, dtype=flint)
        self.cpts_array = [self.cpts] + [None for _ in range(p)]
        self.p = p
        self.t = KnotVector(t)

    def _deboor_1d(self, c: npt.NDArray[flint], x: Num, n: int = 0) -> Point:
        """Perform de Boor's algorithm with arbitrary control points
        @param c The 1-D array of control points
        @param x The parametric point
        @param n Optional reduction of degree for calculation of derivatives
        @return The result of de Boor's algorithm
        """
        k = self.t.k(x)
        p = self.p-n
        q_shape = [p+1] + list(np.shape(c[0]))
        q = np.zeros(q_shape, dtype=flint)
        for i in range(p+1):
            if 0 <= k-p+i < len(c):
                q[i] = c[k-p+i]
        for r in range(p):
            for j in range(p,r,-1):
                l, m = np.clip((j+k-p, j+k-r), 0, len(self.t)-1)
                a = (x-self.t[l])/(self.t[m]-self.t[l])
                q[j] = a*q[j] + (1-a)*q[j-1]
        return q[p]

    def _calc_d_cpts(self, c: list[npt.NDArray[flint]], n: int):
        """Calculate the control points for the derivative spline
        @param n
        """
        if n > self.p:
            raise ValueError("Can only calculate first {self.p} derivative control points")
        if n == 0:
            raise ValueError("Can not calculate 0th order points, must be given")
        else:
            if c[n-1] is None:
                self._calc_d_cpts(c, n-1)
            new_shape = list(c[n-1].shape)
            new_shape[0] += 1
            r = c[n-1].copy()
            r.resize(new_shape)
            p = self.p-(n-1)
            for i in range(new_shape[0]-1,-1,-1):
                dt = self.t[i+p]-self.t[i]
                r_im1 = 0*r[0] if i-1 == -1 else r[i-1]
                r[i] = 0*r[0] if dt == 0 else p*(r[i]-r_im1)/dt
            c[n] = r

    def __call__(self, x: Num) -> Point:
        """Evaluate the spline curve at a parametric point
        @param x The parametric point
        @return The value of the spline at the parametric point x
        """
        return self.d(x, 0)
            
    def d(self, x: Num, n: int = 1) -> Point:
        """Evaluate the n^th order derivative of the spline curve
        @param x The parametric point
        @param n The order of the derivative
        @return The value of the derivative curve at the parametric point x
        """
        out_shape = list(np.shape(x)) + list(np.shape(self.cpts[0]))
        if n > self.p:
            return np.zeros(out_shape, dtype=flint)
        if self.cpts_array[n] is None:
            self._calc_d_cpts(self.cpts_array, n)
        c = self.cpts_array[n]
        out_array = np.zeros(out_shape, dtype=flint)
        with np.nditer(np.array(x), flags=['multi_index']) as it:
            for xx in it:
                out_array[it.multi_index] = self._deboor_1d(c, xx, n)
        return out_array

