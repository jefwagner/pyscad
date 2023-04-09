## @file op.py
"""\
Contains the constructive solid geometry (CSG) operator classes and methods
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


class Op(Csg):
    """CSG Operator abstract base class"""
    children: list[Csg]
 
    def __init__(self, *children: Csg):
        """Create a new Op object"""
        super().__init__()
        self.children = []
        for child in children:
            if not isinstance(child, Csg):
                raise ValueError(f'CSG operators only act CSG objects')
            self.children.append(child)

class Union(Op):
    """A CSG Union operator"""
    ...

class Diff(Op):
    """A CSG Difference operator"""
    ...

class IntXn(Op):
    """A CSG Intersection operator"""
    ...


