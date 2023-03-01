from typing import Union, Sequence, Iterator, List, Callable

import numpy as np
import numpy.typing as npt

from .flint import flint

# Number is a float or floating-point interval
_Num = Union[float, flint]
# And a control point is either a number or list of numbers
CPoint = Union[_Num, Sequence[_Num]]
# We will need arrays of flints, so it helps to vectorize the constructor
v_flint = np.vectorize(flint)


def q0(c: Sequence[CPoint], i: int) -> CPoint:
    """Convenience function for extending a sequence beyond its limits with 0s""" 
    return 0*c[0] if (i < 0 or i >= len(c)) else c[i]    

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
        q = np.array([q0(c, k-r) for r in range(p,-1,-1)])
        for r in range(p):
            for j in range(p,r,-1):
                l, m = j+k-p, j+k-r
                a = (x-self.t[l])/(self.t[m]-self.t[l])
                q[j] = a*q[j] + (1-a)*q[j-1]
        return q[p]

    # def deboor(self, p: int, c: Sequence[CPoint], x: float) -> CPoint:
    #     """Evaluate a b-spline on the knot-vector at a parametric
    #     @param p The degree of the b-spline 
    #     @param c The sequence of control points
    #     @param x The value of parametric argument to the b-spline
    #     @return The D-dimensional point on the b-spline. Note: if x is outside the range
    #     of the knot vector it is evaluated using the b-spline basis polynomial for the
    #     first or last non-zero interval in the knot-vector.
    #     """
    #     # return self.v_deboor(p, c, x)
    #     res = self.v_deboor(p, c, x)
    #     if isinstance(res, np.ndarray):
    #         # # if not isinstance(res, (flint, float)):
    #         if len(res.shape) > 0 and isinstance(res[0], np.ndarray):
    #             return np.vstack(res)
    #     return res

    def d_points(self, c: Sequence[CPoint], p:int, n: int) -> npt.NDArray[CPoint]:
        """Create the new control points for the derivative b-spline
        @param c The sequence of control points
        @param p The degree of the b-spline 
        @param n The order of the derivative
        @return The set of control points for a p-n degree b-spline that is the
        derivative of the p degree b-spline represented by the original control points
        """
        _c = np.array(c)
        r = np.append(_c, [0*_c[0]]*n, axis=0) 
        for k in range(n):
            for i in range(len(r)-n+k,-1,-1):
                dt = self.t[i+p-k]-self.t[i]
                if dt != 0:
                    r[i] = (p-k)*(r[i]-r[i-1])/dt
                else:
                    r[i] = 0*r[i]
        return r


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

    def arc_len(self, a: float, b: float) -> float:
        """Find the arc length along the curve
        @param a The starting parametric value
        @param b The ending parametric value
        @return The arc length of the curve between a and b
        """
        raise NotImplementedError('Need to define a norm for CPoints made of flint objects')
        raise NotImplementedError('Need to define an numerical integration scheme that works with flint objects')

    def tangent(self, x: float) -> CPoint:
        """Find the tangent vector along the curve
        @param x The parametric point
        @return The tangent vector
        """
        return NotImplementedError('Need to define norm for CPoints made of flint objects')

    def curvature(self, x: float) -> CPoint:
        """Find the curvature along the curve
        @param x The parametric point
        @return The curvature
        """
        return NotImplementedError('Need to define norm for CPoints made of flint objects')


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
    
    def __call__(self, x: float) -> CPoint:
        """Evaluate the basis spline
        @param x Parametric
        @return Point along the spline
        """
        return self.t.deboor(self.p, self.c, x)

    def d(self, x: float, n: int = 1) -> CPoint:
        """Evaluate the derivative with respect to the parametric argument
        @param x Parametric value
        @param n The order of the derivative
        @return The value of the derivative at the parametric value x
        """
        d = self.t.d_points(self.p, self.c, n)
        return self.t.deboor(self.p-n, d, x)


# Small number binomimal coefficients for derivative calculations
binom = np.array([
    [1,0,0,0,0],
    [1,1,0,0,0],
    [1,2,1,0,0],
    [1,3,3,1,0],
    [1,4,6,4,1],
])

class NurbsCurve(SpaceCurve):
    """Non-uniform Rational Basis Splines"""

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
                res -= binom[i+1,k]*((s[i+1-k].T*w[k]).T)
            s.append((res.T/w[0]).T)
        return s

    def d(self, x: float, n: int = 1) -> CPoint:
        """Evaluate the derivative of the b-spline with respect
        @param x Parametric value
        @param n The order of the derivative
        @return The n^th derivative at the point x along the spline
        """
        return self.d_list(x, n)[-1]
        

