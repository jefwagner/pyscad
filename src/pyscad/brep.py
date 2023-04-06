"""@file brep.py Boundary Representation

This file is part of pyscad.

pyscad is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version
3 of the License, or (at your option) any later version.

pyscad is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with pyscad. If
not, see <https://www.gnu.org/licenses/>.
"""

from typing import Sequence

import numpy as np

from .surf import ParaSurf

class BRep:
    """A boundary representation (or brep) of a solid object.
    
    The brep describes a solid object as a set of faces, that describe the 2-dimensional
    surfaces of the object, as well as the edges where adjacent faces meet.
    """

    def __init__(self):
        """Create a new empty boundary-rep"""
        self.f = []
        self.e = []


class Edge:
    """An edge in a boundary representation
    
    An edge defines the boundary between two adjacent faces. The edge will define a
    mapping from the 1-D interval [0,1] to the 2-D parameter space for the primary
    surface s0(u,v). The direction between the endpoints goes counterclockwise around
    for the primary surface s0, and clockwise in the secondary surface s1.
    """

    def __init__(self, s0: ParaSurf, s1: ParaSurf):
        """Create a new brep edge object
        @param s0 The primary surface containing the edge
        @param s1 The secondary surface containing the edge
        """
        self.s0 = s0
        self.s1 = s1


class Face:
    """A face in a boundary representation.
    
    A face consist of a surface, and an edge loop. The surface is a parametric surface
    that maps a 2-D (u,v) parameters space to 3-D (x,y,z) Euclidean space. The edge loop
    is an ordered list of edges. Two consecutive edges in the edge loop must share an
    end point. Traversing from edge to edge in edge loop defines the 'outside' of the
    face using the right hand rule.
    """

    def __init__(self, s: ParaSurf, el: Sequence[Edge]):
        """Create a new brep face object
        @param s The surface for the face
        @param el An ordered sequence of edges that define the boundary of the face.
        """
        self.s = s # < Paramatric Surface
        self.el = list(el) # < Sequence of edges
        self.h = []
