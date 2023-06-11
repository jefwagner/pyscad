## @file brep.py
"""\
Contains the boundary representation BRep class and methods
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

from typing import Union, Literal

import numpy as np

from ..utils.types import *
from ..geo import Surface, Curve

EdgeOri = Union[Literal[1], Literal[-1]]

class Edge:
    """A bounded curve"""
    curve: Curve
    p0: Point
    pf: Point

class Face:
    """A simply connected surface with a defined boundary"""
    surf: Surface
    edge_loop: list[tuple[Edge, EdgeOri]]

class BRep:
    """A boundary representation (or brep) of a solid object"""
    faces: list[Face]
