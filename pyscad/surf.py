"""@file surf.py Abstract base class for parametric surfaces

This file is part of pyscad.

pyscad is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version
3 of the License, or (at your option) any later version.

pyscad is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Foobar. If
not, see <https://www.gnu.org/licenses/>.
"""
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .flint import FloatLike
from .cpoint import *

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

    @cp_vectorize
    def normal(self, u: float, v: float) -> CPoint:
        """Evaluate a unit normal vector to the surface
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
        # Do the cross-product to get a vector normal to the surface
        n = cp_cross(eu, ev)
        if cp_mag(n) == 0:
            raise ZeroDivisionError(f'degenerate point at (u,v)=({u},{v})')
        return cp_unit(n)

    def shape_op(self, u: float, v: float) -> npt.NDArray[FloatLike]:
        """Evaluate the shape operator
        @param u The u parameter
        @param v The v parameter
        @return The first fundamental form as an array with the structure 
        [[E, F], [F, G]]
        """
        # Get the partial derivative vectors in the u and v directions
        res = self.d_tri(u,v,2)
        eu = res[1][0]
        ev = res[0][1]
        euu = res[2][0]
        euv = res[1][1]
        evv = res[0][2]
        # Get the normal vector
        n = cp_cross(eu, ev)
        if cp_mag(n) == 0:
            raise ZeroDivisionError(f'Degenerate point at (u,v)=({u},{v}): ds/du and ds/dv are linearly dependent')
        n = cp_unit(n)
        # Take the appropriate dot products
        E = np.dot(eu, eu)
        F = np.dot(eu, ev)
        G = np.dot(ev, ev)
        L = np.dot(euu, n)
        M = np.dot(euv, n)
        N = np.dot(evv, n)
        # Make the fundamental form arrays
        if E*G-F*F <= 0:
            raise ZeroDivisionError(f'First fundamental form has a non-positive determinant')
        fff_inv = np.array([[G,-F],[-F,E]])/(E*G-F*F) # inv of first fund form
        sff = np.array([[L,M],[M,N]]) # second fund form
        # Return the shape operator
        return np.dot(sff, fff_inv)

    @cp_vectorize(ignore=(2,))
    def k_mean(self, 
               u: float, 
               v: float, 
               P: Optional[npt.NDArray[FloatLike]] = None
               ) -> FloatLike:
        """Evaluate the mean surface curvatures
        @param u The u parameter
        @param v The v parameter
        @param P The shape operator for the surface at (u,v)
        @return The mean curvature of the surface at point (u,v)
        """
        if P is None:
            P = self.shape_op(u, v)
        return 0.5*(P[0,0]+P[1,1])
 
    @cp_vectorize(ignore=(2,))
    def k_gaussian(self, 
                   u: float, 
                   v: float, 
                   P: Optional[npt.NDArray[FloatLike]] = None
                   ) -> FloatLike: 
        """Evaluate the Gaussian surface curvature
        @param u The u parameter
        @param v The v parameter
        @param P The shape operator for the surface at (u,v)
        @return The Gaussian curvature of the surface at point (u,v)
        """
        if P is None:
            P = self.shape_op(u, v)
        return P[0,0]*P[1,1]-P[0,1]*P[1,0]

    def k_principal(self, 
                    u: float, 
                    v: float, 
                    P: Optional[npt.NDArray[FloatLike]] = None
                    ) -> npt.NDArray[FloatLike]:
        """Evaluate the principal surface curvatures
        @param u The u parameter
        @param v The v parameter
        @param P The shape operator for the surface at (u,v)
        @return The two principal curvatures of the surface at point (u,v)
        """
        if P is None:
            P = self.shape_op(u, v)
        return cp_2x2eigvals(P)

    def k_princ_vec(self, 
                    u: float, 
                    v: float, 
                    P: Optional[npt.NDArray[FloatLike]] = None
                    ) -> Tuple[npt.NDArray[FloatLike], npt.NDArray[FloatLike]]:
        """Evaluate the principal surface curvatures and their directions in u,v space
        @param u The u parameter
        @param v The v parameter
        @param P The shape operator for the surface at (u,v)
        @return The two principal curvatures of the surface at point (u,v) and the
        vectors in u,v space that give the direction along which a curve has the 
        principle curvatures
        """
        if P is None:
            P = self.shape_op(u, v)
        return cp_2x2eigsys(P)

