## @file test_nurbs.py 
"""\
Validate NURBS behavior.
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

import pytest

import numpy as np
from flint import flint

from pyscad.geo.curve import ParaCurve
from pyscad.geo.spline.kvec import KnotVector
from pyscad.geo.spline.bspline import BSplineCurve
from pyscad.geo.spline.nurbs import NurbsCurve


class TestNurbsCurveInternal:
    """Validate 1-d nurbs curve internal methods"""

    def test_init(self):
        nc = NurbsCurve([1],[1],2,[0,1,2,3])
        assert isinstance(nc, ParaCurve)
        assert isinstance(nc, BSplineCurve)
        assert isinstance(nc, NurbsCurve)
        assert isinstance(nc.w, list)
        assert len(nc.w) == 3
        w = nc.w[0]
        assert isinstance(w, np.ndarray)
        assert w.dtype == flint
