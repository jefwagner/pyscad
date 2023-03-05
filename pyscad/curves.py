from typing import Union, Sequence, Iterator, List, Callable

import numpy as np
import numpy.typing as npt
import scipy.integrate as integrate

from .flint import flint, v_flint
from .cpoint import CPoint, cp_mag, cp_unit, cp_vectorize

class KnotVector:
    """Basis Spline Knot Vector"""

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
        k = np.searchsorted(self.t, x)-1
        return np.clip(k, self.kmin, self.kmax)

    @staticmethod
    def q0(c: Sequence[CPoint], i: int) -> CPoint:
        """Convenience function for extending a sequence beyond its limits with 0s""" 
        return 0*c[0] if (i < 0 or i >= len(c)) else c[i]    

    def deboor(self,c: Sequence[CPoint], p: int,  x: float) -> CPoint:
        """Evaluate a b-spline on the knot-vector at a parametric
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

    @cp_vectorize
    def tangent(self, x: float) -> CPoint:
        """Find the tangent vector along the curve
        @param x The parametric point
        @return The tangent vector
        """
        t = self.d(x, 1)
        return cp_unit(t)
 
    @cp_vectorize
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
        return self._eval(x, 0, self.cpts)

    def d(self, x: float, n: int = 1) -> CPoint:
        """Evaluate the derivative with respect to the parametric argument
        @param x Parametric value
        @param n The order of the derivative
        @return The value of the derivative at the parametric value x
        """
        cpts = self.t.d_cpts(self.c, self.p)
        for i in range(1,n):
            cpts = self.t.d_cpts(d_cpts, self.p-i)
        return self._eval(x, n, cpts)

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
            res.append(self.eval(x, i, cpts))
        return res


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
        

