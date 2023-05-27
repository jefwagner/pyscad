## @file curve.py 
"""\
Space curve base class
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

from ..types import *


class ParaCurve:
    """A parametric space curve for t in R to R^3"""

    def __call__(self, t: Num) -> Point:
        """Evaluate the curve
        @param t The parametric value
        @return The value of the curve
        """
        return self.d(t, 0)

    def d(self, t: Num, n: int = 1) -> Point:
        """Evaluate the n^th order derivative of the curve
        @param t The parametric value
        @param n The order of the derivative
        @return The n^th derivative of the curve
        """
        out_shape = list(np.shape(n)) + list(np.shape(t)) + list(self.shape)
        out_array = np.zeros(out_shape, dtype=flint)
        pre = [np.s_[:]]*len(np.shape(n))
        post = [np.s_[:]]*len(self.shape)
        for idx, tt in np.ndenumerate(t):
            array_idx = tuple(pre + list(idx) + post)
            out_array[array_idx] = self.d_vec(tt, n)
        return out_array

    def d_vec(self, t: Num, n: int) -> Point:
        """Evaluate derivative vectorized over the derivative orders
        @param t The parametric value
        @param n The order of the derivative
        @return The n^th derivative of the curve
        """
        out_shape = list(np.shape(n)) + list(self.shape)
        out_array = np.zeros(out_shape, dtype=flint)
        for idx, nn in np.ndenumerate(n):
            out_array[idx] = self.d_nv(t, nn)
        return out_array

    def d_nv(self, t: Num, n: int) -> Point:
        """Evaluate derivative non-vectorized over the t and n parameters
        @param t The parametric value
        @param n The order of the derivative
        @return The n^th derivative of the curve
        """
        raise NotImplementedError("Virtual method, must redefine")

    def t(self, t: Num) -> Point:
        """Evaluate then tangent of the space curve at a parametric value
        @param t The parametric value
        @return The tangent of the curve
        """
        out_shape = list(np.shape(t)) + list(self.shape)
        out_array = np.zeros(out_shape, dtype=flint)
        for idx, tt in np.ndenumerate(t):
            d = self.d_vec(tt, 1)
            m = mag(d)
            out_array[idx] = d if m == 0 else d/m
        return out_array

    def kap(self, t: Num) -> Point:
        """Evaluate the curvature of the space curve at a parametric value
        @param t The parametric value
        @return The curvature of the curve
        """
        out_shape = list(np.shape(t))
        out_array = np.zeros(out_shape, dtype=flint)
        for idx, tt in np.ndenumerate(t):
            d1, d2 = self.d_vec(tt,[1,2])
            if mag(d1) == 0:
                out = np.zeros(np.shape, dtype=flint)
                if mag(d2) == 0:
                    for x in out:
                        x.a = np.nan
                        x.b = np.nan
                        x.v = np.nan
                else:
                    for x in out:
                        x.a = np.inf
                        x.b = np.inf
                        x.v = np.inf
                out_array[idx] = out
            else:                    
                num = np.cross(d1,d2)
                denom = mag(d1)*mag(d1)*mag(d1)
                out_array[idx] = num/denom
        return out_array


class Line(ParaCurve):
    """Simple line"""

    def __init__(self, p0: Point, p1: Point):
        self.cpts = np.array([p0, p1], dtype=flint)
        self.shape = self.cpts[0].shape

    def d_nv(self, x: Num, n: int = 1) -> Point:
        if n == 0:
            return p0 + (p1-p0)*x
        elif n == 1:
            return (p1-p0)
        else:
            return np.zeros(self.shape, dtype=flint)
