## @file prim.py
"""\
Contains the constructive solid geometry (CSG) primitive classes and methods
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

from .csg import Csg
from ..utils.types import *
from ..utils.trans import Translate, Scale


class Prim(Csg):
    """CSG Primitive abstract base class"""

    def __init__(self, pos: Vec = (0,0,0)):
        super().__init__()
        if not is_vec(pos, length=3):
            raise TypeError("Position must be a vector of length 3")
        self.params.extend(['pos'])
        self.pos = np.array(pos, dtype=flint)
        self.trans = [Translate(self.pos)]

    def __repr__(self) -> str:
        return self.__class__.__name__

class Sphere(Prim):
    """A CSG Sphere primitive"""

    def __init__(self, r: Num = 1, pos: Vec = (0,0,0)):
        """Create a new Sphere object
        @param r The radius of the sphere
        @param pos The position of the center of the sphere
        """
        super().__init__(pos)
        if not is_num(r):
            raise TypeError("Radius must be a number")
        self.params.extend(['r'])
        self.r = flint(r)
        self.trans.insert(0, Scale(self.r))

    def construct_brep(self):
        self._brep = Brep.sphere()

class Box(Prim):
    """A CSG Box primitive"""

    def __init__(self, size: Union[Num, Vec] = 1, pos: Vec = (0,0,0)):
        """Create a new box
        @param size The size of the Box
        @param pos The position of the lower-left-front corner of the Box
        """
        super().__init__(pos)
        if (not is_num(size)) and (not is_vec(size, length=3)):
            raise TypeError("Size must be a number or set of three numbers")
        self.params.extend(['size'])
        if is_num(size):
            self.size = np.array([size, size, size], dtype=flint)
        else:
            self.size = np.array(size, dtype=flint)
        self.trans.insert(0, Scale(self.size))


class Cyl(Prim):
    """A CSG Cylinder primitive"""

    def __init__(self, h: Num = 1, r: Num = 1, pos: Vec = (0,0,0)):
        """Create a new cylinder
        @param h The height or length of the cylinder
        @param r The radius of the cylinder
        @position The position of the lower-center point of the cylinder
        """
        super().__init__(pos)
        if (not is_num(h)) or (not is_num(r)):
            raise TypeError("Height and radius must be numbers")
        self.params.extend(['h','r'])
        self.h = flint(h)
        self.r = flint(r)
        self.trans.insert(0, 
            Scale((self.r, self.r, self.h)))


class Cone(Prim):
    """A CSG Cone primitive"""

    def __init__(self, h = 1, r = 1, pos: Vec = (0,0,0)):
        """Create a new cone
        @param h The height of the cone
        @param r The radius of the cylinder
        @position The position of the lower-center point of the cone
        """
        super().__init__(pos)
        if (not is_num(h)) or (not is_num(r)):
            raise TypeError("Height and radius must be numbers")
        self.params.extend(['h','r'])
        self.h = flint(h)
        self.r = flint(r)
        self.trans.insert(0, 
            Scale((self.r, self.r, self.h)))
