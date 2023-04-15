## @file csg.py
"""\
Contains the constructive solid geometry (CSG) base class and methods
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

from typing import Union, Any, Optional

from .types import *
from .trans import Transform, Scale, Translate, Rotate


class Csg:
    """Constructive Solid Geometry (CSG) abstract base class"""
    trans: list[Transform] # All CSG objects can have their own list of transforms
    params: list[str] # A list of parameter names used for serialization
    meta: dict(str, Any) # Extra metadata can be supplied for each object

    def __init__(self):
        self.trans = []
        self.params = []
        self.meta = {}

    #
    # Method for adding transformations to a CSG object
    #
    # All methods return a copy of self so that they can be chained together
    # using a builder pattern.`
    #
    def scale(self, size: Union[Number, Vec]) -> 'Csg':
        """Apply a scaling transformation"""
        self.trans.append(Scale(size))
        return self

    def trans(self, dx: Vec) -> 'Csg':
        """Apply a translation transformation"""
        self.trans.append(Translate(dx))
        return self

    def rot(self, angle: Num, angle: Optional[Vec] = None) -> 'Csg':
        """Apply a rotation transformation around arbitrary axis"""
        self.trans.append(Rotate(angle, angle))
        return self
    
    def rotx(self, angle: Num) -> 'Csg':
        """Apply a rotation transformation around the x-axis"""
        self.trans.append(Rotate(angle, (1,0,0)))
        return self

    def roty(self, angle: Num) -> 'Csg':
        """Apply a rotation transformation around the y-axis"""
        self.trans.append(Rotate(angle, (0,1,0)))
        return self

    def rotz(self, angle: Num) -> 'Csg':
        """Apply a rotation transformation around the z-axis"""
        self.trans.append(Rotate(angle, (0,0,1)))
        return self

    def rotzxz(self, alpha: Num, beta: Num, gamma: Num) -> 'Csg':
        """Apply rotation transformations using ZXZ euler angles"""
        self.trans.extend([
            Rotate((0,0,1), alpha),
            Rotate((1,0,0), beta),
            Rotate((0,0,1), gamma)
        ])
        return self

    def add_metadata(self, key: str, value: Any) -> 'Csg':
        """Add metadata to the CSG object"""
        self.meta[key] = value
        return self

    def ser(self) -> dict:
        """Build a python dict for JSON serialization"""
        # Start off with object type
        state = dict()
        state['__module__'] = self.__module__
        state['__class__'] = self.__class__.__name__
        # Then Include the list of transforms
        state['trans'] = [t.ser() for t in self.trans]
        # Next add all object specific parameters
        params = {}
        for param_name in self.params:
            param_val = getattr(self, param_name)
            if isinstance(param_val, np.ndarray):
                param_ser = array_ser(param_val)
            else:
                param_ser = num_ser(param_val)
            params[p] = param_ser
        state['params'] = params
        # # Finally include meta-data
        # state['meta'] = self.meta
        return state

        

