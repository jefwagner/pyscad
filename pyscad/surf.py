"""@file Differential geometry of curves
"""

import numpy as np
import numpy.typing as npt

from .cpoint import CPoint, cp_unit

class ParaSurf:
    """A parametric surface from u,v to R^3"""

    def __call__(self, u: float, v: float) -> CPoint:
        raise NotImplementedError("Virtual method, must redefine")

    def d(self, u: float, v: float, nu: int, nv: int) -> CPoint:
        raise NotImplementedError("Virtual method, must redefine")

    def normal(self, u: float, v: float) -> CPoint:
        """Return a uit normal vector to the surface
        @param u The u parameter
        @param v The v parameter
        @return A unit length normal vector for the surface at the parametric point 
        (u,v).
        Note: This method does NOT handle degenerate points on a surface
        """
        # Get the derivative vectors in the u and v directions
        eu = self.d(u,v,1,0)
        ev = self.d(u,v,0,1)
        # Do the cross-product by hand
        n = np.array([
            eu[1]*ev[2]-eu[2]*ev[1],
            eu[2]*ev[0]-eu[0]*ev[1],
            eu[0]*ev[1]-eu[0]*ev[1],
        ], dtype=eu.dtype)
        return cp_unit(n)

