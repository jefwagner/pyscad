"""@file flint.py Floating point rounded interval arithmetic
Implements a new numeric class for floating point intervals

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

# Most operations will act on intervals for normal floats
from typing import Union, Sequence

# We will use the numpy implementation of IEEE 754 64 bit floats
import numpy as np

class flint:
    a: np.float64
    b: np.float64
    v: np.float64

    def __init__(self, val: Union[float, 'flint']):
        '''Create a new interval object'''
        if isinstance(val, flint):
            self.a = val.a
            self.b = val.b
            self.v = val.v
        else:
            self.a = np.float64(val)
            self.b = np.float64(val)
            self.v = np.float64(val)

    def __float__(self) -> float:
        '''Cast the interval to a float'''
        return float(self.v)

    def __repr__(self) -> str:
        '''Build a string representation of the interval'''
        return f'{self.v}'

    @staticmethod
    def identical(first: 'flint', second: 'flint') -> bool:
        '''Compare two intervals to check if they are identical'''
        return first.v == second.v and first.a == second.a and first.b == second.b

    @classmethod
    def from_interval(cls, a: float, b: float) -> 'flint':
        '''Directly create a new interval object'''
        x = flint(0)
        x.a = np.float64(a)
        x.b = np.float64(b)
        x.v = np.float64(0.5*(a+b))
        return x

    @classmethod
    def frac(cls, num: float, denom: float) -> 'flint':
        '''Create a interval from a fraction'''
        result = flint(num/denom)
        result._grow()
        return result

    def _grow(self):
        '''Expand the interval by the 'units in last place' for min an max values'''
        self.a = np.nextafter(self.a, -np.inf)
        self.b = np.nextafter(self.b, np.inf)

    def __eq__(self, other: Union[float, 'flint']) -> bool:
        other = flint(other)
        return (self.a <= other.b) and (self.b >= other.a)

    def __ne__(self, other: Union[float, 'flint']) -> bool:
        other = flint(other)
        return (self.a > other.b) or (self.b < other.a)

    def __le__(self, other: Union[float, 'flint']) -> bool:
        other = flint(other)
        return self.a <= other.b

    def __lt__(self, other: Union[float, 'flint']) -> bool:
        other = flint(other)
        return self.b < other.a

    def __ge__(self, other: Union[float, 'flint']) -> bool:
        other = flint(other)
        return self.b >= other.a

    def __gt__(self, other: Union[float, 'flint']) -> bool:
        other = flint(other)
        return self.a > other.b

    def __add__(self, other: Union[float, 'flint']) -> 'flint':
        other = flint(other)
        result = flint(0)
        result.v = self.v + other.v
        result.a = self.a + other.a
        result.b = self.b + other.b
        result._grow()
        return result

    __radd__ = __add__

    def __iadd__(self, other: Union[float, 'flint']) -> 'flint':
        other = flint(other)
        self.v += other.v
        self.a += other.a
        self.b += other.b
        self._grow()
        return self

    def __sub__(self, other: Union[float, 'flint']) -> 'flint':
        other = flint(other)
        result = flint(0)
        result.v = self.v - other.v
        result.a = self.a - other.b
        result.b = self.b - other.a
        result._grow()
        return result

    def __rsub__(self, other: float) -> 'flint':
        other = flint(other)
        return other.__sub__(self)

    def __isub__(self, other: Union[float, 'flint']) -> 'flint':
        other = flint(other)
        self.v -= other.v
        self.a -= other.b
        self.b -= other.a
        self._grow()
        return self

    def __mul__(self, other: Union[float, 'flint']) -> 'flint':
        other = flint(other)
        vals = [self.a*other.a, self.a*other.b, self.b*other.a, self.b*other.b]
        result = flint(0)
        result.v = self.v*other.v
        result.a = np.min(vals)
        result.b = np.max(vals)
        result._grow()
        return result
    
    __rmul__ = __mul__

    def __imul__(self, other: Union[float, 'flint']) -> 'flint':
        other = flint(other)
        vals = [self.a*other.a, self.a*other.b, self.b*other.a, self.b*other.b]
        self.v *= other.v
        self.a = np.min(vals)
        self.b = np.max(vals)
        self._grow()
        return self

    def __truediv__(self, other: Union[float, 'flint']) -> 'flint':
        other = flint(other)
        vals = [self.a/other.a, self.a/other.b, self.b/other.a, self.b/other.b]
        result = flint(0)
        result.v = self.v/other.v
        result.a = np.min(vals)
        result.b = np.max(vals)
        result._grow()
        return result

    def __rtruediv__(self, other: float) -> 'flint':
        other = flint(other)
        return other.__truediv__(self)

    def __itruediv__(self, other: Union[float, 'flint']) -> 'flint':
        other = flint(other)
        vals = [self.a/other.a, self.a/other.b, self.b/other.a, self.b/other.b]
        self.v /= other.v
        self.a = np.min(vals)
        self.b = np.max(vals)
        self._grow()
        return self

    def __neg__(self) -> 'flint':
        result = flint(0)
        result.a = -self.b
        result.b = -self.a
        result.v = -self.v
        return result

    def __pos__(self) -> 'flint':
        return flint(self)

    def __abs__(self) -> 'flint':
        result = flint(0)
        if self.a < 0 and self.b < 0:
            result.a = -self.b
            result.b = -self.a
            result.v = -self.v
        elif self.a < 0 and self.b >= 0:
            result.a = np.float64(0)
            result.b = np.max([-self.a, self.b])
            result.v = -self.v if self.v < 0 else self.v
        else:
            result.a = self.a
            result.b = self.b
            result.v = self.v
        return result

    def sqrt(self) -> 'flint':
        result = flint(0)
        if self.b < 0.0:
            raise ValueError('math domain error')
        result.a = 0.0 if self.a < 0 else np.nextafter(np.sqrt(self.a), -np.inf)
        result.v = 0.0 if self.v < 0 else np.sqrt(self.v)
        result.b = np.nextafter(np.sqrt(self.b), np.inf)
        return result

# A number is either a float or floating point interval
FloatLike = Union[float, flint]

# We will need arrays of flints, so it helps to vectorize the constructor
_v_flint = np.vectorize(flint)
# This little function allows numbers to remain numbers
def v_flint(x: FloatLike) -> flint:
    """A ufunc that cast all elements to flints"""
    ret = _v_flint(x)
    return ret if ret.shape != () else ret.item()
