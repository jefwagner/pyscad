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

from ..types import *

class ParaCurve:
    """A parametric space curve for t in R to R^3"""

    def __call__(self, t: Num) -> Point:
        raise NotImplementedError("Virtual method, must redefine")

    def d(self, t: Num, n: int = 1) -> Point:
        raise NotImplementedError("Virtual method, must redefine")

    def t(self, t: Num) -> Point:
        """Evaluate then tangent of the space curve at a parametric value
        @param t The parametric value
        @return The tangent of the curve
        """
        d = self.d(t)
        return d/mag(d)[...,np.newaxis]

    def kap(self, t: Num) -> Point:
        """Evaluate the curvature of the space curve at a parametric value
        @param t The parametric value
        @return The curvature of the curve
        """
        t = np.array(t, dtype=flint)
        d1 = self.d(t)
        d2 = self.d(t,2)
        cr = np.cross(d1,d2)
        denom = mag(d1)*mag(d1)*mag(d1)
        if t.shape == cr.shape:
            num = cr
            return num/denom
        else:
            num = mag(cr)
            return num/denom[...,np.newaxis]
