## @file trans.py 
"""\
Defines the affine transform classes to act on 2D and 3D vectors.
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

from typing import Union, Optional
import numpy.typing as npt

import numpy as np
from flint import flint

from .types import *


class Transform:
    """Affine Transform"""
    m: npt.NDArray[flint]
    v: npt.NDArray[flint]

    def __init__(self, dim: int = 3):
        """Create a new transform"""
        if dim not in [2,3]:
            raise ValueError("Transforms must act on 2 or 3 dimensional vectors")
        self.m = np.eye(dim, dtype=flint)
        self.v = np.zeros(dim, dtype=flint)

    def __len__(self):
        """The dimension of vectors the transform act on"""
        return len(self.v)

    @property
    def dim(self):
        """The dimension of vectors the transform acts on"""
        return len(self.v)

    def __call__(self, a: Union[Vec,'Transform']) -> Union[Vec,'Transform']:
        """Apply a transformation"""
        if isinstance(a, Transform):
            t = Transform(len(a))
            t.m = self.m.dot(a.m)
            t.v = self.m.dot(a.v) + self.v
            return t
        return self.m.dot(a) + self.v

    def __eq__(self, other: 'Transform') -> bool:
        """Compare if two transformations are equivalent"""
        if not isinstance(other, Transform):
            raise TypeError("Can only compare transforms")
        if len(self.v) != len(other.v):
            return False
        return np.alltrue(self.m == other.m) and np.alltrue(self.v == other.v)

    def json(self) -> dict:
        """Build a python dict for JSON serialization"""
        trans_dict = dict(name=self.__class__.__name__)
        

class Scale(Transform):
    """Scale"""

    def __init__(self, s: Union[Num, Vec]):
        """Create a new Scale transformations
        @param s If s is a scalar then create a uniform 3D scaling
        transformation, else if s is a vector then scales the each axis
        according to the vectors components. The only way to obtain a 2D uniform
        scaling is to pass in 2D vector with equal components. 
        """
        if is_num(s):
            self.m = np.array([[s,0,0],[0,s,0],[0,0,s]], dtype=flint)
            self.v = np.zeros(3, dtype=flint)
        elif is_vec(s):
            self.m = np.diag(s).astype(flint)
            self.v = np.zeros(len(s), dtype=flint)
        else:
            raise TypeError("Scale must be set with a scalar or vector")


class Translate(Transform):
    """Translate"""

    def __init__(self, dx: Vec):
        if is_vec(dx):
            self.m = np.eye(len(dx), dtype=flint)
            self.v = np.array(dx, dtype=flint)
        else:
            raise TypeError("Translate must be set with a vector")


class Rotate(Transform):
    """Rotation"""

    def __init__(self, angle: Num, axis: Optional[Vec] = None):
        if not is_num(angle):
            raise TypeError("Angle must be a number")
        th = flint(angle)
        if axis is None:
            self.m = np.array([
                [np.cos(th), -np.sin(th)],
                [np.sin(th), np.cos(th)]
            ], dtype=flint)
            self.v = np.zeros(2, dtype=flint)
        elif is_vec(axis) and len(axis) == 3:
            u = np.array(axis, dtype=flint)
            x, y, z = u/np.sqrt(np.sum(u*u))
            c, s = np.cos(th), np.sin(th)
            self.m = np.array([
                [c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                [y*z*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
                [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]
            ], dtype=flint)
            self.v = np.zeros(3, dtype=flint)
        else:
            raise TypeError("Axis must be a 3 dimensional vector")
