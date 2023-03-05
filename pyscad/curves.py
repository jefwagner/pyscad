from typing import Sequence, List

import numpy as np
import numpy.typing as npt
import scipy.integrate as integrate

from .flint import v_flint
from .cpoint import CPoint, cp_mag, cp_unit, cp_vectorize
from .kvec import KnotVector

class SpaceCurve:
    """An abstract base class for D-dimensional parametric curve
    This class assumes that the corresponding curve class will have a vectorized
    `__call__(x)` function that gives the value at a parametric point `x` and a
    vectorized derivative function `d(x,n)` that gives the `n`^th derivative with
    respect to the parametric parameter at the parametric point `x`.
    """

    def __call__(self, x: float) -> CPoint:
        raise NotImplementedError('Virtual method, must redefine')

    def d(self, x: float, n: int) -> CPoint:
        raise NotImplementedError('Virtual method, must redefine')

    def d_list(self, x: float, n: int) -> List[CPoint]:
        """Return the value and first n derivatives of the curve
        @param x The parametric value
        @param n The highest order of the derivative
        @return a list containing the value of the curve and its first n derivativs at 
        the parametric point x
        """
        dl = [self.__call__(x)]
        for i in range(n):
            dl.append(self.d(x,i+1))
        return dl

    def arclen(self, a: float, b: float) -> float:
        """Find the arc length along the curve
        @param a The starting parametric value
        @param b The ending parametric value
        @return The arc length of the curve between a and b
        Note: this cast all flint intervals into floats before doing the integration so
        the result is only approximate. This should not matter, since the arc length is
        evaluated using a numerical approximation anyway.
        """
        res = integrate.quad(
            lambda t: np.sqrt(
                np.sum(
                    (self.d(t,1).astype(float))**2, 
                    axis=-1
                )
            ),a,b)
        return res[0]

    def tangent(self, x: float) -> CPoint:
        """Find the tangent vector along the curve
        @param x The parametric point 
        @return The tangent vector
        """
        t = self.d(x, 1)
        return cp_unit(t)
 
    def curvature(self, x: float) -> CPoint:
        """Find the curvature along the curve
        @param x The parametric point
        @return The curvature
        """
        _, t, n = self.d_list(x, 2)
        sh = list(t.shape)[:-1]
        num = 1
        for dim in sh:
            num *= dim
        c = np.cross(t,n)
        cmag = c.reshape((num,))
        for i in range(num):
            cmag[i] = cp_mag(cmag[i])
        cmag = cmag.reshape(sh)
        tmag = cp_mag(t)
        return cmag/(tmag*tmag*tmag)


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
        self.w = np.array(w, dtype=np.float64)
        self.p = p
        self.t = KnotVector(t)
    
    def __call__(self, x: float) -> CPoint:
        """Evaluate the Nurbs curve
        @param x Parametric value
        @return Point along the spline
        """
        wc = (self.c.T*self.w).T
        c = self.t.deboor(self.p, wc, x)
        w = self.t.deboor(self.p, self.w, x)
        return (c.T/w).T

    def d_list(self, x: float, n: int = 1) -> List[CPoint]:
        """Evaluate the value and derivatives of the Nurbs curvs
        @param x Parametric value
        @param n The order of the derivative
        @return A list of the value and higher order derivatives of the spline curve
        """
        c, w, s = [], [], []
        wc = (c.T*w).T
        _w = self.w.copy()
        c.append(self.t.deboor(self.p, wc, x))
        w.append(self.t.deboor(self.p, _w, x))
        s.append((c[0].T/w[0]).T)
        for i in range(n):
            wc = self.t.d_points(self.p-i, wc, 1)
            _w = self.t.d_points(self.p-i, _w, 1)
            c.append(self.t.deboor(self.p-i-1, wc, x))
            w.append(self.t.deboor(self.p-i-1, _w, x))
            # calc the next derivative
            res = c[-1]
            for k in range(1,i+2):
                res -= self._binom[i+1,k]*((s[i+1-k].T*w[k]).T)
            s.append((res.T/w[0]).T)
        return s

    def d(self, x: float, n: int = 1) -> CPoint:
        """Evaluate the derivative of the b-spline with respect
        @param x Parametric value
        @param n The order of the derivative
        @return The n^th derivative at the point x along the spline
        """
        return self.d_list(x, n)[-1]
        

