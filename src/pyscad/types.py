## @file types.py 
"""\
Defines Num, Vec, and Point as generic number, vector, and point types and test
for those types.
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

from typing import Union, Any, Sequence, Optional, Tuple
import numpy.typing as npt

import functools
import numpy as np
from flint import flint

np_numbers = (np.int8, np.uint8, np.int16, np.uint16, 
              np.int32, np.uint32, np.int64, np.uint64, 
              np.float32, np.float64)

Num = Union[float, flint]
Num.__doc__ = """Generic number as float or flint to be used for type hints"""
def is_num(x: Any) -> bool:
    """Test if input is a generic number
    @param x The input to test
    @return True if x is an integer, float, or flint
    """
    return isinstance(x, (int, float, *np_numbers, flint))

Vec = Sequence[Num]
Vec.__doc__ = """Generic vector as sequence of numbers to be used for type hints"""
def is_vec(v: Any, length: Optional[int] = None) -> bool:
    """Test if input is a generic vector
    @param v The input to test
    @param length [optional] The required length of the vector
    @return True if v is a sequence of numbers of appropriate length
    """
    try:
        a = (len(v) in [2,3]) if length is None else (len(v) == length)
        if not a:
            return False
        if isinstance(v, np.ndarray):
            return v.dtype in (*np_numbers, flint)
        return functools.reduce(lambda a, b: a and b, map(is_num, v))
    except:
        return False

Point = Union[Num, Vec]
Point.__doc__ = """Generic point as number or vector to be used for type hints"""
def is_point(a: Any) -> bool:
    """Test if input is a generic point
    @param a The input to test
    @return True if a is a number or 1, 2, or 3 length vector
    """
    return is_num(a) or is_vec(a) or is_vec(a, length=1)

# Convenience functions for turning a flint/number or array into a JSON
# serializable format.
def num_json(x: Num) -> Union[float, dict]:
    """Convert a generic number into a JSON serializable object"""
    if isinstance(x, flint):
        return {
            'a': float(x.a), 
            'b': float(x.b), 
            'v': float(x.v)
        }
    elif isinstance(x, (int, float, *np_numbers)):
        return x
    else:
        raise TypeError('Can only convert general number type')

def array_json(a: npt.NDArray[Num]) -> list:
    """Convert a numpy array into a JSON serializable object"""
    if not isinstance(a, np.ndarray) or a.dtype not in (*np_numbers, flint):
        raise TypeError('Can only convert NumPy ndarrays of numbers')
    if len(a.shape) == 1:
        return [num_json(x) for x in a]
    else:
        return [array_json(suba) for suba in a]

