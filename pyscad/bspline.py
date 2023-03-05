"""@file Basis spline curves and surfaces
"""
from typing import Sequence, List

import numpy as np
import numpy.typing as npt

from .flint import v_flint
from .cpoint import CPoint, cp_vectorize
from .kvec import KnotVector
from .curves import SpaceCurve

class BSpline(SpaceCurve):
    """Normalized Basis Splines"""

    def __init__(self, c: Sequence[CPoint], p: int, t: Sequence[float]):
        """Create a new b-spline object
        @param c The control points
        @param p Degree of the b-spline basis functions
        @param t the knot-vector
        """
        if len(t) != len(c) + p + 1:
            raise ValueError('Knot vector wrong length')
        self.c = v_flint(np.array(c))
        self.p = p
        self.t = KnotVector(t)
    
    @cp_vectorize(ignore=(1,2))
    def _eval(self, x: float, n: int, cpts: npt.NDArray[CPoint]) -> CPoint:
        """Vectorized internal method for evaluating points and derivatives
        @param cpts The control points to use for the evaluation
        @param n The order of the derivative
        @param x The parametric point
        @return The value of spline or the nth derivative at the point x
        """
        return self.t.deboor(cpts, self.p-n, x)

    def __call__(self, x: float) -> CPoint:
        """Evaluate the basis spline
        @param x Parametric
        @return Point along the spline
        """
        return self.t.deboor(self.c, self.p, x)

    def d(self, x: float, n: int = 1) -> CPoint:
        """Evaluate the derivative with respect to the parametric argument
        @param x Parametric value
        @param n The order of the derivative
        @return The value of the derivative at the parametric value x
        """
        cpts = self.t.d_cpts(self.c, self.p)
        for i in range(1,n):
            cpts = self.t.d_cpts(d_cpts, self.p-i)
        return self.t.deboor(cpts, self.p-n, x)

    def d_list(self, x: float, n: int = 1) -> List[CPoint]:
        """Evaluate the derivative with respect to the parametric argument
        @param x Parametric value
        @param n The highest order of the derivative
        @return a list containing the value of the bspline and its first n derivatives 
        at the parametric point x
        """
        d_cpts_list = self.t.d_cpts_list(self.c, self.p, n)
        res = []
        for i, cpts in enumerate(d_cpts_list):
            res.append(self.t.deboor(cpts, self.p-i, x))
        return res

