## @file nurbs.py 
"""
"""
# Copyright (c) 2023, Jef Wagner <jefwagner@gmail.com>
#
# This file is part of pyscad.
#
# pyscad is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pyscad is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pyscad. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from flint import flint

class NurbsCurve:
    """Non-Uniform Rational Basis Spline
    
    :param verts: A reference to a Nx3 numpy array of vertices
    :param cpts: The list of control points as indices into the verts array
    :param w: The list of weights for the control points
    :param kv: The splines knot-vector
    """

    def __init__(self):
        self.verts = verts
        self.cpts = np.array(cpts, dtype=np.int16)
        self.w = np.array(w, dtype=flint)
        self.kv = np.array(kv, dtype=flint)

    @property
    def deg(self) -> int:
        """The degree of the spline basis functions"""
        return self.kv.shape[0]-self.cpts.shape[0]-1
