"""@file Non-Uniform Rational Basis Spline (NURBS) curves and surfaces
"""
from typing import Sequence, List

import numpy as np
import numpy.typing as npt

from .flint import v_flint
from .cpoint import CPoint, cp_vectorize
from .kvec import KnotVector
from .curves import SpaceCurve

class NurbsCurve(SpaceCurve):
    """Non-uniform Rational Basis Splines"""

    binom = np.array([
        [1,0,0,0,0],
        [1,1,0,0,0],
        [1,2,1,0,0],
        [1,3,3,1,0],
        [1,4,6,4,1],
    ])

    def __init__(self, 
                 c: Sequence[CPoint], 
                 w: Sequence[float], 
                 p: int, 
                 t: Sequence[float]):
        """Create a new NURBS curve object
        @param c The control points
        @param w The weights
        @param p Degree of the b-spline basis functions
        @param t the knot-vector
        """
        if len(c) != len(w):
            raise ValueError('The control points and weights must have the same length')
        if len(t) != len(c) + p + 1:
            raise ValueError('Knot vector wrong length')
        self.c = v_flint(c)
        self.w = np.array([[wi] for wi in w], dtype=np.float64)
        self.p = p
        self.t = KnotVector(t)
    
    def __call__(self, x: float) -> CPoint:
        """Evaluate the Nurbs curve
        @param x Parametric value
        @return Point along the spline
        """
        wc = self.c*self.w
        c = self.t.deboor(self.p, wc, x)
        w = self.t.deboor(self.p, self.w, x)
        return c/w

    def d_list(self, x: float, n: int = 1) -> List[CPoint]:
        """Evaluate the value and derivatives of the Nurbs curvs
        @param x Parametric value
        @param n The order of the derivative
        @return A list of the value and higher order derivatives of the spline curve
        """
        c, w, s = [], [], []
        wc = self.c*self.w
        _w = self.w.copy()
        c.append(self.t.deboor(self.p, wc, x))
        w.append(self.t.deboor(self.p, _w, x))
        s.append(c[0]/w[0])
        for i in range(n):
            wc = self.t.d_points(self.p-i, wc, 1)
            _w = self.t.d_points(self.p-i, _w, 1)
            c.append(self.t.deboor(self.p-i-1, wc, x))
            w.append(self.t.deboor(self.p-i-1, _w, x))
            # calc the next derivative
            res = c[-1]
            for k in range(1,i+2):
                res -= self._binom[i+1,k]*(s[i+1-k]*w[k])
            s.append(res/w[0])
        return s

    def d(self, x: float, n: int = 1) -> CPoint:
        """Evaluate the derivative of the b-spline with respect
        @param x Parametric value
        @param n The order of the derivative
        @return The n^th derivative at the point x along the spline
        """
        return self.d_list(x, n)[-1]
        
