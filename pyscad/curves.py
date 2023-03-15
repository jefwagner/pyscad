"""@file curves.py Abstract base class for parametric curves

This file is part of pyscad.

pyscad is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version
3 of the License, or (at your option) any later version.

pyscad is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Foobar. If
not, see <https://www.gnu.org/licenses/>.
"""
from typing import Sequence, List

import numpy as np
import numpy.typing as npt
import scipy.integrate as integrate

from .cpoint import CPoint, cp_mag, cp_unit

class ParaCurve:
    """An abstract base class for D-dimensional parametric curve
    This class assumes that the corresponding curve class will have a vectorized
    `__call__(x)` function that gives the value at a parametric point `x` and a
    vectorized derivative function `d(x,n)` that gives the `n`^th derivative with
    respect to the parametric parameter at the parametric point `x`.
    """

    def __call__(self, x: float) -> CPoint:
        raise NotImplementedError('Virtual method, must redefine')

    def d(self, x: float, n: int) -> CPoint:
        raise NotImplementedError('Virtual method, must redefine')

    def d_list(self, x: float, n: int) -> List[CPoint]:
        """Return the value and first n derivatives of the curve
        @param x The parametric value
        @param n The highest order of the derivative
        @return a list containing the value of the curve and its first n derivatives at 
        the parametric point x
        """
        dl = [self.__call__(x)]
        for i in range(n):
            dl.append(self.d(x,i+1))
        return dl

    def arclen(self, a: float, b: float) -> float:
        """Find the arc length along the curve
        @param a The starting parametric value
        @param b The ending parametric value
        @return The arc length of the curve between a and b
        Note: this cast all flint intervals into floats before doing the integration so
        the result is only approximate. This should not matter, since the arc length is
        evaluated using a numerical approximation anyway.
        """
        res = integrate.quad(
            lambda t: np.sqrt(
                np.sum(
                    (self.d(t,1).astype(float))**2, 
                    axis=-1
                )
            ),a,b)
        return res[0]

    def tangent(self, x: float) -> CPoint:
        """Find the tangent vector along the curve
        @param x The parametric point 
        @return The tangent vector
        """
        t = self.d(x, 1)
        return cp_unit(t)
 
    def curvature(self, x: float) -> CPoint:
        """Find the curvature along the curve
        @param x The parametric point
        @return The curvature
        """
        _, t, n = self.d_list(x, 2)
        sh = list(t.shape)[:-1]
        num = 1
        for dim in sh:
            num *= dim
        c = np.cross(t,n)
        cmag = c.reshape((num,))
        for i in range(num):
            cmag[i] = cp_mag(cmag[i])
        cmag = cmag.reshape(sh)
        tmag = cp_mag(t)
        return cmag/(tmag*tmag*tmag)
