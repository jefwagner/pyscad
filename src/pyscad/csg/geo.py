from typing import Union, Any, Sequence, Optional
import numpy as np
import numpy.typing as npt

import collections.abc
import functools

from flint import flint

Num = Union[float, flint]
Num.__doc__ = """Generic number as float or flint to be used for type hints"""
def is_num(x: Any) -> bool:
    """Test if input is a generic number"""
    return isinstance(x, (int, float, flint))

Vec = Sequence[Num]
Vec.__doc__ = """Generic vector as sequence of numbers to be used for type hints"""
def is_vec(v: Any, length: Optional[int] = None) -> bool:
    """Test if input is a generic vector"""
    try:
        a = (len(v) in [1, 2,3]) if length is None else (len(v) == length)
        if isinstance(v, np.ndarray):
            return a and v.dtype in (int, float, flint)
        return a and functools.reduce(lambda a, b: a and b, map(is_num, v))
    except:
        return False

Point = Union[Num, Vec]
Point.__doc__ = """Generic point as number or vector to be used for type hints"""
def is_point(a: Any) -> bool:
    """Test if input is a generic point"""
    return is_num(a) or is_vec(a)


class Transform:
    """Affine Transform"""
    m: npt.NDArray[flint]
    v: npt.NDArray[flint]

    def __init__(self, dim: int = 3):
        """Create a new transform"""
        if dim not in [2,3]:
            raise ValueError()
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
            t.m = self.m.dot(a.m) + self.v.dot(a.m)
            t.v = self.m.dot(a.v) + self.v
            return t
        return self.m.dot(a) + self.v

    def __eq__(self, other: 'Transform') -> bool:
        if not isinstance(other, Transform):
            raise ValueError("Can only compare transforms")
        if len(self.v) != len(other.v):
            return False
        return np.alltrue(self.m == other.m) and np.alltrue(self.v == other.v)


class Scale(Transform):
    """Scale"""

    def __init__(self, s: Union[Num, Vec]):
        if is_num(s):
            self.m = np.array([[s,0,0],[0,s,0],[0,0,s]], dtype=flint)
            self.v = np.zeros(3, dtype=flint)
        elif is_vec(s):
            self.m = np.diag(s, dtype=flint)
            self.v = np.zeros(len(s), dtype=flint)
        else:
            raise ValueError("Scale must be set with a scalar or vector")


class Translate(Transform):
    """Translate"""

    def __init__(self, dx):
        ...

class Rotate(Transform):
    """Rotation"""

    def __init__(self, axis, angle):
        ...
