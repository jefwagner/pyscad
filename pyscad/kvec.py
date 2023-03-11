"""@file One and two dimensional knot vectors for evaluation of basis spline curves and
surfaces.
"""
from typing import Sequence, Iterator, List, Literal

import numpy as np
import numpy.typing as npt

from .cpoint import CPoint, cp_vectorize

class KnotVector:
    """Basis spline knot vector"""

    def __init__(self, t: Sequence[float]):
        """Create a new knot-vector
        @param t the knot-vector
        """
        # Validate the knot-vector is non-decreasing
        low = t[0]
        for high in t[1:]:
            if low > high:
                raise ValueError('The knot vector must be a sequence of only non-decreasing values')
            low = high
        self.t = np.array(t, dtype=np.float64)
        # Get the min and max non-zero length interval indexes
        self.kmin = 0
        for tt in t[1:]:
            if tt != t[0]:
                break
            self.kmin += 1
        self.kmax = len(t)-2
        for tt in t[-2::-1]:
            if tt != t[-1]:
                break
            self.kmax -= 1
        # # Vectorize the de-Boor's algorithm method
        # self.v_deboor = np.vectorize(self.s_deboor, excluded=[0,1])
  
    def __len__(self) -> int:
        """Length of the knot vector"""
        return len(self.t)

    def __getitem__(self, i: int) -> float:
        """Get the i^th knot"""
        return self.t[i]

    def __iter__(self) -> Iterator[float]:
        """Iterate over the knots"""
        return iter(self.t)

    def k(self, x: float) -> int:
        """Find the index of the interval containing the parametric argument
        @param x The parameter - should be between t_min and t_max 
        @return The index of the interval in the knot-vector containing the parametric
        argument x. Note: If x is outside the range of the knot-vector, then this will
        return the index of the first or last non-zero interval in the knot-vector
        """
        k = np.searchsorted(self.t, x, side='right')-1
        return np.clip(k, self.kmin, self.kmax)

    @staticmethod
    def q0(c: Sequence[CPoint], i: int) -> CPoint:
        """Convenience function for extending a sequence beyond its limits with 0s""" 
        return 0*c[0] if (i < 0 or i >= len(c)) else c[i]    

    def deboor_nv(self, c: Sequence[CPoint], p: int,  x: float) -> CPoint:
        """Evaluate a b-spline on the knot-vector at a parametric Non-Vectorized
        @param c The sequence of control points
        @param p The degree of the b-spline 
        @param x The value of parametric argument to the b-spline
        @return The D-dimensional point on the b-spline. Note: if x is outside the range
        of the knot vector it is evaluated using the b-spline basis polynomial for the
        first or last non-zero interval in the knot-vector.
        """
        k = self.k(x)
        q = np.array([self.q0(c, k-r) for r in range(p,-1,-1)])
        for r in range(p):
            for j in range(p,r,-1):
                l, m = np.clip((j+k-p, j+k-r),0,len(self.t)-1)
                a = (x-self.t[l])/(self.t[m]-self.t[l])
                q[j] = a*q[j] + (1-a)*q[j-1]
        return q[p]

    @cp_vectorize(ignore=(0,1))
    def deboor(self, c: Sequence[CPoint], p: int,  x: float) -> CPoint:
        """Evaluate a b-spline on the knot-vector at a parametric
        @param c The sequence of control points
        @param p The degree of the b-spline 
        @param x The value of parametric argument to the b-spline
        @return The D-dimensional point on the b-spline. Note: if x is outside the range
        of the knot vector it is evaluated using the b-spline basis polynomial for the
        first or last non-zero interval in the knot-vector.
        """
        return self.deboor_nv(c, p, x)

    def d_cpts(self, c: Sequence[CPoint], p: int) -> npt.NDArray[CPoint]:
        """Create the new control points for the first derivative b-spline
        @param c The sequence of control points
        @param p The degree of the b-spline 
        @return The set of control points for a p-1 degree b-spline that is the
        derivative of the p degree b-spline represented by the original control points
        """
        _c = np.array(c)
        r = np.append(_c, [0*_c[0]], axis=0) 
        for i in range(len(r)-1,-1,-1):
            dt = self.t[i+p]-self.t[i]
            r_im1 = r[i-1] if i-1 != -1 else 0*r[0]
            if dt != 0:
                r[i] = (p)*(r[i]-r_im1)/dt
            else:
                r[i] = 0*r[i]
        return r

    def d_cpts_list(self, c: Sequence[CPoint], p: int, n: int) -> List[npt.NDArray[CPoint]]:
        """
        @param c The sequence of control points 
        @param p The degree of the b-spline
        @param n The order of the derivative
        @return A list of numpy ndarrays of control points that define the
        original control points and all the  
        """
        pts_list = [np.array(c)]
        for i in range(n):
            pts_list.append(self.d_cpts(pts_list[-1], p-i))
        return pts_list


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

    @cp_vectorize(ignore=(0,1,2))
    def deboor(self, 
               c: Sequence[Sequence[CPoint]],
               pu: int, 
               pv: int, 
               u: float, 
               v: float) -> CPoint:
        """Evaluate a direct product b-spline surface
        @param c The 2-d array of control points for a spline surface
        @param pu The degree of the spline in the u-direction
        @param pv The degree of the spline in the v-direction
        @param u The u parameter
        @param v The v parameter
        @return The value of the surface at the parametric point (u,v)
        """
        c = np.array(c)
        cj = np.empty_like(c[:,0])
        for i in range(len(c)):
            cj[i] = self.tv.deboor_nv(c[i], pv, v)
        return self.tu.deboor_nv(cj, pu, u)

    def du_cpts(self, c: Sequence[Sequence[CPoint]], pu: int) -> npt.NDArray[CPoint]:
        """Create the new control points for the u partial derivative of the b-spline
        surface
        @param c The 2-D array of control points
        @param pu The degree of the b-spline in the u direction
        @return The set of control points for a pu-1
        """
        _c = np.array(c)
        n, m = _c.shape[0], _c.shape[1]
        r = np.append(_c, [[0*_c[0,0]]*m], axis=0)
        for i in range(n,-1,-1):
            dt = self.tu[i+pu]-self.tu[i]
            r_im1 = r[i-1,:] if i-1 != -1 else 0*r[0,:]
            if dt != 0:
                r[i,:] = (pu)*(r[i,:]-r_im1)/dt
            else:
                r[i,:] = 0*r[i,:]
        return r

    def dv_cpts(self, c: Sequence[Sequence[CPoint]], pv: int) -> npt.NDArray[CPoint]:
        """Create the new control points for the v partial derivative of the b-spline 
        surface
        @param c The 2-D array of control points
        @param pv The degree of the b-spline in the v direction
        @return The set of control points for a pu,pv-1 b-spline surface
        """
        _c = np.array(c)
        n, m = _c.shape[0], _c.shape[1]
        r = np.append(_c, [[0*_c[0,0]]]*n, axis=1)
        for i in range(m,-1,-1):
            dt = self.tv[i+pv]-self.tv[i]
            r_im1 = r[:,i-1] if i-1 != -1 else 0*r[:,0]
            if dt != 0:
                r[:,i] = (pv)*(r[:,i]-r_im1)/dt
            else:
                r[:,i] = 0*r[:,i]
        return r

    def d_cpts_rect(self, 
                    c: Sequence[Sequence[CPoint]], 
                    pu: int, 
                    pv: int, 
                    nu: int,
                    nv: int) -> List[List[npt.NDArray[CPoint]]]:
        """Create an rectangular array of new control points for partial deriviatives of
        the b-spline surface
        @param c The 2-D array of control points
        @param pu The degree of the b-spline in the u direction
        @param pv The degree of the b-spline in the v direction
        @param nu The order of the 'u' partial derivative
        @param nv The order of the 'v' partial derivative
        @return A (nu+1, nv+1) rectangular array of control points        
        """
        cpts = [[None for _ in range(nu+1)] for _ in range(nv+1)]
        cpts[0][0] = np.array(c)
        for j in range(1, nv+1):
            cpts[0][j] = self.dv_cpts(cpts[0][j-1], pv-(j-1))
        for i in range(1, nu+1):
            for j in range(nv+1):
                cpts[i][j] = self.du_cpts(cpts[i-1][j], pu-(i-1))
        return cpts

    def d_cpts_tri(self, 
                   c: Sequence[Sequence[CPoint]], 
                   pu: int, 
                   pv: int, 
                   nmax: int) -> List[List[npt.NDArray[CPoint]]]:
        """Create an triangular array of new control points for partial deriviatives of
        the b-spline surface
        @param c The 2-D array of control points
        @param pu The degree of the b-spline in the u direction
        @param pv The degree of the b-spline in the v direction
        @param nmax The total maximum order of the partial derivatives
        @return A nmax+1 sized triangular array of control points        
        """
        cpts = [[None for _ in range(nmax+1-i)] for i in range(nmax+1)]
        cpts[0][0] = np.array(c)
        for j in range(1, nmax+1):
            cpts[0][j] = self.dv_cpts(cpts[0][j-1], pv-(j-1))
        for i in range(1, nmax+1):
            for j in range(0, nmax+1-i):
                cpts[i][j] = self.du_cpts(cpts[i-1][j], pu-(i-1))
        return cpts
        