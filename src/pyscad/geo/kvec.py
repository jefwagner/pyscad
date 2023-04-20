## @file kvec.py 
"""\
One and two dimensional knot vectors
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

from typing import Sequence, Iterator
import numpy.typing as npt

import numpy as np
from flint import flint

from ..types import *

class KnotVector:
    """Basis spline knot vector"""

    def __init__(self, t: Sequence[Num]):
        """Create a new knot-vector
        @param t the knot-vector
        """
        # Validate the knot-vector is non-decreasing
        low = t[0]
        for high in t[1:]:
            if low > high:
                raise ValueError('The knot vector must be a sequence of only non-decreasing values')
            low = high
        self.kmin = 0
        # Get the min and max non-zero length interval indices
        for tt in t[1:]:
            if tt != t[0]:
                break
            self.kmin += 1
        self.kmax = len(t)-2
        for tt in t[-2::-1]:
            if tt != t[-1]:
                break
            self.kmax -= 1
        # Set the knot-vector
        self.t = np.array(t, dtype=flint)

    def __len__(self) -> int:
        """Length of the knot vector"""
        return len(self.t)

    def __getitem__(self, i: int) -> float:
        """Get the i^th knot"""
        return self.t[i]

    def __iter__(self) -> Iterator[float]:
        """Iterate over the knots"""
        return iter(self.t)

    def k(self, x: Num) -> int:
        """Find the index of the interval containing the parametric argument
        @param x The parameter - should be between t_min and t_max 
        @return The index of the interval in the knot-vector containing the parametric
        argument x. Note: If x is outside the range of the knot-vector, then this will
        return the index of the first or last non-zero interval in the knot-vector
        """
        k = np.searchsorted(self.t, x, side='right')-1
        return np.clip(k, self.kmin, self.kmax)
        
    @staticmethod
    def q0(c: npt.NDArray[Point], i: int) -> Point:
        """Convenience function for extending a sequence beyond its limits with 0s""" 
        return 0*c[0] if (i < 0 or i >= len(c)) else c[i]    

    def deboor(self, c: npt.NDArray[Point], p: int,  x: Num) -> Point:
        """Evaluate a b-spline on the knot-vector at a parametric Non-Vectorized
        @param c The sequence of control points
        @param p The degree of the b-spline 
        @param x The value of parametric argument to the b-spline
        @return The D-dimensional point on the b-spline. Note: if x is outside the range
        of the knot vector it is evaluated using the b-spline basis polynomial for the
        first or last non-zero interval in the knot-vector.
        """
        k = self.k(x)
        q = np.array([self.q0(c, k-r) for r in range(p,-1,-1)])
        for r in range(p):
            for j in range(p,r,-1):
                l, m = np.clip((j+k-p, j+k-r),0,len(self.t)-1)
                a = (x-self.t[l])/(self.t[m]-self.t[l])
                q[j] = a*q[j] + (1-a)*q[j-1]
        return q[p]


class KnotMatrix:
    """2 knot vectors used in a direct product b-spline surface"""

    def __init__(self, tu: Sequence[Num], tv: Sequence[Num]):
        """Create a new knot-matrix object
        @param tu The knot-vector in the u-direction
        @param tv The knot-vector in the v-direction
        """
        self.tu = KnotVector(tu)
        self.tv = KnotVector(tv)
        self.shape = (len(self.tu), len(self.tv))

    def deboor(self, c: npt.NDArray[Point],
               pu: int, pv: int, u: Num, v: Num) -> Point:
        """Evaluate a direct product b-spline surface
        @param c The 2-d array of control points for a spline surface
        @param pu The degree of the spline in the u-direction
        @param pv The degree of the spline in the v-direction
        @param u The u parameter
        @param v The v parameter
        @return The value of the surface at the parametric point (u,v)
        """
        c = np.array(c)
        cj = np.empty_like(c[:,0])
        for i in range(len(c)):
            cj[i] = self.tv.deboor(c[i], pv, v)
        return self.tu.deboor(cj, pu, u)
