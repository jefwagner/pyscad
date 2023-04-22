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

from ...types import *

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

    def __getitem__(self, i: int) -> flint:
        """Get the i^th knot"""
        return self.t[i]

    def __iter__(self) -> Iterator[flint]:
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

