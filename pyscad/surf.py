from typing import Union, Sequence, Iterator, List, Callable

import numpy as np
import numpy.typing as npt

from .flint import flint
from .curves import KnotVector

# Number is a float or floating-point interval
_Num = Union[float, flint]
# And a control point is either a number or list of numbers
CPoint = Union[_Num, Sequence[_Num]]
# We will need arrays of flints, so it helps to vectorize the constructor
v_flint = np.vectorize(flint)

class KnotMatrix:
    """2 knot vectors used in a direct product b-spline surface"""

    def __init__(self, tu: Sequence[float], tv: Sequence[float]):
        """Create a new knot-matrix object
        @param tu The knot-vector in the u-direction
        @param tv The knot-vector in the v-direction
        """
        self.tu = KnotVector(tu)
        self.tv = KnotVector(tv)
        self.shape = (len(self.tu), len(self.tv))

    def deboor(self, 
               c: Sequence[Sequence[CPoint]],
               pu: int, 
               pv: int, 
               u: float, 
               v: float) -> CPoint:
        """Evaluate a direct product b-spline surface
        @param c The 2-d array of control points for the surface
        @param pu The degree of the spline in the u-direction
        @param pv The degree of the spline in the v-direction
        @param u The u parameter
        @param v The v parameter
        @return The value at the parametric point (u,v)
        """
        c = np.array(c)
        cj = np.empty_like(c[:,0])
        for i in range(len(c)):
            cj[i] = self.tv.deboor(pv, c[i], v)
        return self.tu.deboor(pu, cj, u)


class ParaSurf:
    """A parametric surface from u,v to R^3"""

    def __call__(self, u: float, v: float) -> CPoint:
        raise NotImplementedError("Virtual method, must redefine")

    def d(self, u: float, v: float, nu: int, nv: int) -> CPoint:
        raise NotImplementedError("Virtual method, must redefine")

    def normal(self, u: float, v: float) -> CPoint:
        eu = self.d(u,v,1,0)
        ev = self.d(u,v,0,1)
        n = np.array([
            eu[1]*ev[2]-eu[2]*ev[1],
            eu[2]*ev[0]-eu[0]*ev[1],
            eu[0]*ev[1]-eu[0]*ev[1],
        ], dtype=object)
        nmag = flint.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        return n/nmag

class NurbsSurf(ParaSurf):

    def __init__(self,
                 c: Sequence[Sequence[flint]],
                 w: Sequence[Sequence[float]],
                 pu: int,
                 pv: int,
                 tu: Sequence[float],
                 tv: Sequence[float]):
        """Create a new NURBS surface object
        @param c The control points
        @param w The weights
        @param pu Degree of the u direction b-spline basis functions
        @param pv Degree of the v direction b-spline basis functions
        @param tu the u direction knot-vector
        @param tv the v direction knot-vector
        """
        self.c = v_flint(c)
        self.w = np.array(w, dtype=np.float64)
        if self.c.shape[:2] != self.w.shape:
            raise ValueError('The control points and weights must have the same shape')
        if len(tu) != len(c[0]) + pu + 1:
            raise ValueError('u-direction knot vector wrong length')
        if len(tv) != len(c) + pv + 1:
            raise ValueError('v-direction knot vector wrong length')
        self.pu = pu
        self.pv = pv
        self.t = KnotMatrix(tu, tv)

    def __call__(self, u: float, v: float) -> CPoint:
        """Evaluate the surface at a parametric point (u,v)
        @param u the u parameter
        @param v the v parameter
        @return The position of the surface at the parametric point (u,v)
        """
        wc = np.empty_like(self.c)
        for i in range(len(wc)):
            for j in range(len(wc[0])):
                wc[i,j] = self.w[i,j]*self.c[i,j]
        c = self.t.deboor(wc, self.pu, self.pv, u, v)
        w = self.t.deboor(self.w, self.pu, self.pv, u, v)
        return c/w
