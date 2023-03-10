"""@file Differential geometry of curves
"""
from typing import List

import numpy as np
import numpy.typing as npt

from .cpoint import CPoint, cp_unit

class ParaSurf:
    """A parametric surface from u,v to R^3"""

    def __call__(self, u: float, v: float) -> CPoint:
        raise NotImplementedError("Virtual method, must redefine")

    def d(self, u: float, v: float, nu: int, nv: int) -> CPoint:
        raise NotImplementedError("Virtual method, must redefine")

    def d_rect(self, u: float, v: float, nu: int, nv: int) -> List[List[CPoint]]:
        """Evaluate the surface function and partial derivatives up to a max order
        @param u The u parameter 
        @param v The v parameter
        @param nu The order of the u partial derivative
        @param nv The order of the v partial derivative
        @return A triangular array of the surface and its partial derivatives
        evaluated at the parametric point (u,v). 
        """
        return [[self.d(u,v,i,j) for j in range(nv+1)] for i in range(nu+1)]

    def d_tri(self, u: float, v: float, nmax: int) -> List[List[CPoint]]:
        """Evaluate the surface function and partial derivatives up to a max order
        @param u The u parameter 
        @param v The v parameter
        @param nmax A triangular array of the surface and its partial derivatives
        evaluated at the parametric point (u,v). 
        """
        return [[self.d(u,v,i,j) for j in range(nmax+1-i)] for i in range(nmax+1)]

    def normal(self, u: float, v: float) -> CPoint:
        """Return a uit normal vector to the surface
        @param u The u parameter
        @param v The v parameter
        @return A unit length normal vector for the surface at the parametric point 
        (u,v).
        Note: This method does NOT handle degenerate points on a surface
        """
        # Get the derivative vectors in the u and v directions
        res = self.d_tri(u,v,1)
        eu = res[1][0]
        ev = res[0][1]
        # Do the cross-product by hand
        n = np.array([
            eu[1]*ev[2]-eu[2]*ev[1],
            eu[2]*ev[0]-eu[0]*ev[1],
            eu[0]*ev[1]-eu[0]*ev[1],
        ], dtype=eu.dtype)
        if cp_mag(n) == 0:
            raise ZeroDivisionError(f'degenerate point at (u,v)=({u},{v})')
        return cp_unit(n)

