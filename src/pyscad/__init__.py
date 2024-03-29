## @file pyscad/__init__.py 
"""\
PySCAD: Python based programmers CSG modeling a-la OpenSCAD.
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

from .types import *
from .trans import Transform, Scale, Translate, Rotate
from .csg.prim import Sphere, Box, Cyl, Cone
from .csg.op import Union, Diff, IntXn

# # Export floating point interval objects and the v_flint helper function
# from .flint import flint, v_flint
# # Export the ParaCurve abstract base class
# from .curves import ParaCurve
# # Export the ParaSurf abstract base class
# from .surf import ParaSurf
# # Export the BSpline curve object
# from .bspline import BSpline, BSplineSurf
# # Export the Nurbs curve object
# from .nurbs import NurbsCurve, NurbsSurf

# # Exposing knot vectors and matrix for testing
# from .kvec import KnotMatrix, KnotVector
