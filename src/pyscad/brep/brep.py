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

import numpy as np

from ..types import *
from ..geo import ParaCurve, ParaSurf
from .edge import Edge
from .face import Face

from ..geo.spline import NurbsSurf

class BRep:
    """A boundary representation (or brep) of a solid object"""
    surfs: list[ParaSurf]
    curves: list[ParaCurve]
    verts: list[Point]
    faces: list[Face]
    edges: list[Edge]

    def __init__(self):
        """Create a new empty boundary representation"""
        self.surfs = []
        self.curves = []
        self.verts = []
        self.faces = []
        self.edges = []

    @classmethod
    def sphere(cls):
        c = np.array([
            [[1,0,0],[1,1,0],[0,1,0]],
            [[1,0,1],[1,1,1],[0,1,1]],
            [[0,0,1],[0,0,1],[0,0,1]],
        ], dtype=flint)
        a = 1/np.sqrt(flint(2))
        w = [[1,_a,1],[_a,_a*_a,_a],[1,_a,1]]
        t = [0,0,0,1,1,1]
        m = np.array([[0,1,0],[-1,0,0],[0,0,1]], type=flint)
        z = np.zeros((3,), dtype=flint)
        rot90 = Transform.from_arrays(m, z)
        m = np.array([[1,0,0],[0,1,0],[0,0,-1]], dtype=flint)
        reflz = Transform.from_arrays(m, z)
        sppp = NurbsSurf(c, w, 2, 2, t, t)
        for idx in np.ndindex((3,3)):
            c[idx] = rot90(c[idx])
        snpp = NurbsSurf(c, w, 2, 2, t, t)
        for idx in np.ndindex((3,3)):
            c[idx] = rot90(c[idx])
        snnp = NurbsSurf(c, w, 2, 2, t, t)
        for idx in np.ndindex((3,3)):
            c[idx] = rot90(c[idx])
        spnp = NurbsSurf(c, w, 2, 2, t, t)
        for idx in np.ndindex((3,3)):
            c[idx] = rot90(c[idx])
            c[idx] = reflz(c[idx])
        sppn = NurbsSurf(c, w, 2, 2, t, t)
        for idx in np.ndindex((3,3)):
            c[idx] = rot90(c[idx])
        snpn = NurbsSurf(c, w, 2, 2, t, t)
        for idx in np.ndindex((3,3)):
            c[idx] = rot90(c[idx])
        snnn = NurbsSurf(c, w, 2, 2, t, t)
        for idx in np.ndindex((3,3)):
            c[idx] = rot90(c[idx])
        spnn = NurbsSurf(c, w, 2, 2, t, t)
        self.surfs.extend([
            sppp,
            snpp,
            snnp,
            spnp,
            sppn,
            snpn,
            snnn,
            spnn
        ])

