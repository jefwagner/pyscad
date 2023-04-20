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
    """A parametric surface from t in [0,1] to R^3"""

    def __call__(self, t: Num) -> Point:
        raise NotImplementedError("Virtual method, must redefine")

    def d(self, t: Num, n: int) -> Point:
        raise NotImplementedError("Virtual method, must redefine")
