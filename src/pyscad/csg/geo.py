from typing import Union, Sequence

from flint import flint

Num = Union[float, flint]

Vec = Sequence[Num]

CPoint = Union[Num, Vec]

class Transform:
    """Transform"""
    m: npt.NDArray[flint]
    v: npt.NDArray[flint]

    def __init__(self, dim: int = 3):
        self.m = np.eye(dim, dtype=flint)
        self.v = np.zeros(dim, dtype=flint)

    def __len__(self):
        return len(self.v)

    def __call__(self, a):
        """Apply a transformation"""
        if isinstance(a, Transform):
            t = Transform(len(a))
            t.m = self.m.dot(a.m) + self.v.dot(a.m)
            t.v = self.m.dot(a.v) + self.v
            return t
        return m.dot(a) + v

class Scale(Transform):
    """Scale"""

    def __init__(self, s):
        ...

class Translate(Transform):
    """Translate"""

    def __init__(self, dx):
        ...

class Rotate(Transform):
    """Rotation"""

    def __init__(self, axis, angle):
        ...
