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
        self.c = np.array(c, dtype=flint)
        self.p = p
        self.t = KnotVector(t)

    def _deboor_1d(self, c: npt.NDArray[flint], x: Num) -> Point:
        """Perform de Boor's algorithm with arbitrary control points
        @param c The 1-D array of control points
        @param x The parametric point
        @return The result of de Boor's algorithm
        """
        k = self.t.k(x)
        q_shape = [self.p+1] + list(np.shape(c[0]))
        q = np.zeros(q_shape, dtype=flint)
        for i in range(self.p+1):
            if 0 <= k-self.p+i < len(c):
                q[i] = c[k-self.p+i]
        for r in range(self.p):
            for j in range(self.p,r,-1):
                l, m = np.clip((j+k-self.p, j+k-r), 0, len(self.t)-1)
                a = (x-self.t[l])/(self.t[m]-self.t[l])
                q[j] = a*q[j] + (1-a)*q[j-1]
        return q[self.p]

    def __call__(self, x: Num) -> Point:
        """Evaluate the spline curve at a parametric point
        @param x The parametric point
        @return The value of the spline at the parametric point x
        """
        out_shape = list(np.shape(x)) + list(np.shape(self.c[0]))
        out_array = np.zeros(out_shape, dtype=flint)
        with np.nditer(x, flags=['multi_index']) as it:
            for xx in it:
                out_array[it.multi_index] = self._deboor_1d(self.c, xx)
        return out_array
            
