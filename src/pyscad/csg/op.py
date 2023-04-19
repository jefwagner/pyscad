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
from ..brep import BRep


class Op(Csg):
    """CSG Operator abstract base class"""
    children: list[Csg]
    aop = '#'
    uop = '\u220E'

    def __init__(self, *children: Csg):
        """Create a new Op object"""
        super().__init__()
        self.children = []
        if len(children) < 2:
            raise TypeError(f'Operators require at least 2 operands')
        for child in children:
            if not isinstance(child, Csg):
                raise TypeError(f'CSG operators only act CSG objects')
            self.children.append(child)

    @staticmethod
    def wrap(x) -> str:
        """"""
        return f'({x})' if isinstance(x, Op) else f'{x}'

    def __repr__(self) -> str:
        """"""
        rep = self.wrap(self.children[0])
        for child in self.children[1:]:
            rep += f' {self.aop} {self.wrap(child)}'
        return rep


class Union(Op):
    """A CSG Union operator"""
    aop = 'U'
    uop = '\u222A'

class IntXn(Op):
    """A CSG Intersection operator"""
    aop = 'X'
    uop = '\u2229'

class Diff(Op):
    """A CSG Difference operator"""
    aop = '\\'
    uop = '\u2216'

    def __repr__(self) -> str:
        """"""
        if len(self.children) == 2:
            c0, c1 = self.children
            rep = f'{self.wrap(c0)} {self.aop} {self.wrap(c1)}'
        else:
            c0 = f'{self.wrap(self.children[0])}'
            c1 = f'({self.wrap(self.children[1])}'
            for child in self.children[2:]:
                c1 += f' {Union.aop} {self.wrap(child)}'
            c1 += ')'
            rep = f'{c0} {self.aop} {c1}'
        return rep
