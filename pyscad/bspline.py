"""@file Basis spline curves and surfaces
"""
from typing import Sequence, List

import numpy as np
import numpy.typing as npt

from .flint import v_flint
from .cpoint import CPoint
from .kvec import KnotVector, KnotMatrix
from .curves import ParaCurve
from .surf import ParaSurf

class BSpline(ParaCurve):
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
            cpts = self.t.d_cpts(cpts, self.p-i)
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


class BSplineSurf(ParaSurf):
    """Direct product surface of normalized basis splines"""

    def __init__(self, 
                 c: Sequence[Sequence[CPoint]], 
                 pu: int, 
                 pv: int, 
                 tu: Sequence[float], 
                 tv: Sequence[float]):
        """Create a new b-spline surface object
        @param c The 2-d array of control points
        @param pu The degree of the spline in the u direction
        @param pv The degree of the spline in the v direction
        @param tu The knot-vector in the u direction
        @param tv The knot-vector in the v direction
        """
        self.c = v_flint(np.array(c))
        if len(tu) != len(c[0]) + pu + 1:
            raise ValueError('u-direction knot vector wrong length')
        if len(tv) != len(c) + pv + 1:
            raise ValueError('v-direction knot vector wrong length')
        self.pu = pu
        self.pv = pv
        self.t = KnotMatrix(tu, tv)

    def __call__(self, u: float, v: float) -> CPoint:
        """Evaluate the surface at the parametric point (u,v)
        @param u The u parameter
        @param v The v parameter
        @return A point on the surface corresponding to the parametric point (u,v)
        """
        return self.t.deboor(self.c, self.pu, self.pv, u, v)

    def d(self, u:float, v:float, nu: int, nv: int):
        """Evaluate a partial derivative of the surface the parametric point (u,v)
        @param u The u parameter
        @param v The v parameter
        @param nu The order of the u partial derivative
        @param nv The order of the v partial derivative
        @return A partial derivative of the surface at the parametric point (u,v)
        """
        cpts = self.c
        for i in range(nu):
            cpts = self.t.du_cpts(cpts, self.pu-i)
        for j in range(nv):
            cpts = self.t.dv_cpts(cpts, self.pv-j)
        return self.t.deboor(cpts, self.pu-nu, self.pv-nv, u, v)

    def d_rect(self, u:float, v:float, nu: int, nv: int) -> List[List[CPoint]]:
        """Evaluate the surface function and partial derivatives up to a max order
        @param u The u parameter 
        @param v The v parameter
        @param nu The order of the u partial derivative
        @param nv The order of the v partial derivative
        @return A triangular array of the surface and its partial derivatives
        evaluated at the parametric point (u,v). 
        """
        cpts = self.t.d_cpts_rect(self.c, self.pu, self.pv, nu, nv)
        res = [[None for _ in range(nv+1)] for _ in range(nu+1)]
        for i in range(nu+1):
            for j in range(nv+1):
                res[i][j] = self.t.deboor(cpts[i][j], self.pu-i, self.pv-j, u, v)
        return res

    def d_tri(self, u:float, v:float, nmax: int) -> List[List[CPoint]]:
        """Evaluate the surface function and partial derivatives up to a max order
        @param u The u parameter 
        @param v The v parameter
        @param nmax A triangular array of the surface and its partial derivatives
        evaluated at the parametric point (u,v). 
        """
        cpts = self.t.d_cpts_tri(self.c, self.pu, self.pv, nmax)
        res = [[None for _ in range(nmax+1-i)] for i in range(nmax+1)]
        for i in range(nmax+1):
            for j in range(nmax+1-i):
                res[i][j] = self.t.deboor(cpts[i][j], self.pu-i, self.pv-j, u, v)
        return res
