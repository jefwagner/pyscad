"""@file Non-Uniform Rational Basis Spline (NURBS) curves and surfaces

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
from typing import Sequence, List

import numpy as np
import numpy.typing as npt

from .flint import v_flint
from .cpoint import CPoint, cp_vectorize
from .kvec import KnotVector, KnotMatrix
from .curves import ParaCurve
from .surf import ParaSurf

class NurbsCurve(ParaCurve):
    """Non-uniform Rational Basis Splines"""

    _binom = np.array([
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
        c = self.t.deboor(wc, self.p, x)
        w = self.t.deboor(self.w, self.p, x)
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
        c.append(self.t.deboor(wc, self.p, x))
        w.append(self.t.deboor(_w, self.p, x))
        s.append(c[0]/w[0])
        for i in range(n):
            wc = self.t.d_cpts(wc, self.p-i)
            _w = self.t.d_cpts(_w, self.p-i)
            c.append(self.t.deboor(wc, self.p-i-1, x))
            w.append(self.t.deboor(_w, self.p-i-1, x))
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


class NurbsSurf(ParaSurf):

    _binom = np.array([
        [1,0,0,0,0],
        [1,1,0,0,0],
        [1,2,1,0,0],
        [1,3,3,1,0],
        [1,4,6,4,1],
    ])

    def __init__(self,
                 c: Sequence[Sequence[CPoint]],
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
        sh = list(self.c.shape)[:2]
        self.w = np.array(w, dtype=np.float64).reshape(sh+[1])
        if self.c.shape[:2] != self.w.shape[:2]:
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
        wc = self.c*self.w
        c = self.t.deboor(wc, self.pu, self.pv, u, v)
        w = self.t.deboor(self.w, self.pu, self.pv, u, v)
        return c/w

    def d(self, u:float, v: float, nu: int, nv: int) -> CPoint:
        """Evaluate the nu^th u and nv^th v partial derivative of the surface
        @param u the u parameter
        @param v the v parameter
        @param nu The order of the u partial derivative
        @param nv The order of the v partial derivative
        @return A partial derivative of the surface at the parametric point (u,v)
        """
        res = self.d_rect(u, v, nu, nv)
        return res[nu][nv]

    def d_rect(self, u: float, v: float, nu: int, nv: int) -> List[List[CPoint]]:
        """Evaluate the surface an all partial partial derivatives for i<nu and j<nv
        @param u The u parameter 
        @param v The v parameter
        @param nu The order of the u partial derivative
        @param nv The order of the v partial derivative
        @return A (nu+1, nv+1) rectangular array of the surface and its partial 
        derivatives evaluated at the parametric point (u,v). 
        """
        cpts = self.t.d_cpts_rect(self.w*self.c, self.pu, self.pv, nu, nv)
        wpts = self.t.d_cpts_rect(self.w, self.pu, self.pv, nu, nv)
        c = [[None for _ in range(nv+1)] for _ in range(nu+1)]
        w = [[None for _ in range(nv+1)] for _ in range(nu+1)]
        s = [[None for _ in range(nv+1)] for _ in range(nu+1)]
        c[0][0] = self.t.deboor(cpts[0][0], self.pu, self.pv, u, v)
        w[0][0] = self.t.deboor(wpts[0][0], self.pu, self.pv, u, v)
        s[0][0] = c[0][0]/w[0][0]
        for j in range(1, nv+1):
            c[0][j] = self.t.deboor(cpts[0][j], self.pu, self.pv-j, u, v)
            w[0][j] = self.t.deboor(wpts[0][j], self.pu, self.pv-j, u, v)
            s[0][j] = c[0][j]
            for jj in range(1, j+1):
                s[0][j] -= self._binom[j, jj]*s[0][j-jj]*w[0][jj]
            s[0][j] /= w[0][0]
        for i in range(1, nu+1):
            for j in range(nv+1):
                c[i][j] = self.t.deboor(cpts[i][j], self.pu-i, self.pv-j, u, v)
                w[i][j] = self.t.deboor(wpts[i][j], self.pu-i, self.pv-j, u, v)
                s[i][j] = c[i][j]
                for ii in range(1, i+1):
                    for jj in range(j+1):
                        term = self._binom[i, ii]*self._binom[j, jj]
                        term *= s[i-ii][j-jj]*w[ii][jj]
                        s[i][j] -= term
                s[i][j] /= w[0][0]
        return s

    def d_tri(self, u: float, v: float, nmax: int) -> List[List[CPoint]]:
        """Evaluate the surface function and partial derivatives up to a max order
        @param u The u parameter 
        @param v The v parameter
        @param nmax The maximum total order for partial derivatives (nu+nv <= nmax)
        @return A triangular array of the surface and its partial derivatives
        evaluated at the parametric point (u,v). 
        """
        cpts = self.t.d_cpts_tri(self.w*self.c, self.pu, self.pv, nmax)
        wpts = self.t.d_cpts_tri(self.w, self.pu, self.pv, nmax)
        c = [[None for _ in range(nmax+1-i)] for i in range(nmax+1)]
        w = [[None for _ in range(nmax+1-i)] for i in range(nmax+1)]
        s = [[None for _ in range(nmax+1-i)] for i in range(nmax+1)]
        c[0][0] = self.t.deboor(cpts[0][0], self.pu, self.pv, u, v)
        w[0][0] = self.t.deboor(wpts[0][0], self.pu, self.pv, u, v)
        s[0][0] = c[0][0]/w[0][0]
        for j in range(1, nmax+1):
            c[0][j] = self.t.deboor(cpts[0][j], self.pu, self.pv-j, u, v)
            w[0][j] = self.t.deboor(wpts[0][j], self.pu, self.pv-j, u, v)
            s[0][j] = c[0][j]
            for jj in range(1, j+1):
                s[0][j] -= self._binom[j, jj]*s[0][j-jj]*w[0][jj]
            s[0][j] /= w[0][0]
        for i in range(1, nmax+1):
            for j in range(nmax+1-i):
                c[i][j] = self.t.deboor(cpts[i][j], self.pu-i, self.pv-j, u, v)
                w[i][j] = self.t.deboor(wpts[i][j], self.pu-i, self.pv-j, u, v)
                s[i][j] = c[i][j]
                for ii in range(1, i+1):
                    for jj in range(j+1):
                        term = self._binom[i, ii]*self._binom[j, jj]
                        term *= s[i-ii][j-jj]*w[ii][jj]
                        s[i][j] -= term
                s[i][j] /= w[0][0]
        return s
