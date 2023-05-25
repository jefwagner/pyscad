## @file bspline.py 
"""\
Basis spline curves and surfaces
"""
# Copyright (c) 2023, Jef Wagner <jefwagner@gmail.com>
#
# This file is part of pyscad.
#
# pyscad is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pyscad is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pyscad. If not, see <https://www.gnu.org/licenses/>.

from typing import Sequence
import numpy.typing as npt

import numpy as np
from flint import flint

from ...types import *
from ..curve import ParaCurve
from ..surf import ParaSurf
from .kvec import KnotVector

class BSplineCurve(ParaCurve):
    """Basis Spline"""

    def __init__(self, c: Sequence[Point], p: int, t: Sequence[Num]):
        """Create a new b-spline object
        @param The control points
        @param p Degree of the b-spline basis functions
        @param t The knot-vector
        """
        if len(t) != len(c) + p + 1:
            raise ValueError("Knot vector wrong length for control points")
        self.p = p
        self.t = KnotVector(t)
        self.cpts = np.array(c, dtype=flint)
        self.shape = self.cpts[0].shape
        self.cpts_array = []
        self.calc_cpts_array(self.cpts, self.cpts_array)

    def calc_cpts_array(self, c: npt.NDArray[flint], arr: list[npt.NDArray[flint]]):
        """Calculate the control points for the derivative spline
        @param c The control points
        @param arr The list to hold the derivative control points
        """
        arr.append(c)
        shape = list(c.shape)
        for n in range(self.p):
            shape[0] += 1
            r = np.zeros(shape, dtype=flint)
            r[:-1] = arr[-1][:]
            p = self.p-n
            for i in range(shape[0]-1,0,-1):
                dt = self.t[i+p]-self.t[i]
                r[i] = 0*r[0] if dt == 0 else p*(r[i]-r[i-1])/dt
            dt = self.t[p]-self.t[0]
            r[0] = 0*r[0] if dt == 0 else p*r[0]/dt
            arr.append(r)

    def _deboor_1d(self, c: npt.NDArray[flint], x: Num, n: int = 0) -> Point:
        """Perform de Boor's algorithm with arbitrary control points
        @param c The 1-D array of control points
        @param x The parametric point
        @param n Optional reduction of degree for calculation of derivatives
        @return The result of de Boor's algorithm
        """
        k = self.t.k(x)
        p = self.p-n
        q_shape = [p+1] + list(np.shape(c[0]))
        q = np.zeros(q_shape, dtype=flint)
        for i in range(p+1):
            if 0 <= k-p+i < len(c):
                q[i] = c[k-p+i]
        for r in range(p):
            for j in range(p,r,-1):
                l, m = np.clip((j+k-p, j+k-r), 0, len(self.t)-1)
                a = (x-self.t[l])/(self.t[m]-self.t[l])
                q[j] = a*q[j] + (1-a)*q[j-1]
        return q[p]

    def d_nv(self, x: Num, n: int = 1) -> Point:
        """Evaluate the n^th order derivative of the spline curve
        @param x The parametric point
        @param n The order of the derivative
        @return The value of the derivative curve at the parametric point x
        """
        if n > self.p:
            return np.zeros(self.shape, dtype=flint)
        else:
            return self._deboor_1d(self.cpts_array[n], x, n)


class BSplineSurf(ParaSurf):
    """Direct product direct product spline"""

    def __init__(self, 
                 c: Sequence[Sequence[Point]], 
                 pu: int,
                 pv: int,
                 tu: KnotVector,
                 tv: KnotVector):
        """
        @param c The 2-D array of control points
        @param pu The degree of the b-spline u function
        @param pv The degree of the b-spline v function
        @param tu The u direction knot-vector
        @param tv The v direction knot-vector
        """
        super().__init__()
        if len(tu) != len(c) + pu + 1:
            raise ValueError("Knot vector wrong length for control points")
        if len(tv) != len(c[0]) + pv + 1:
            raise ValueError("Knot vector wrong length for control points")
        self.pu = pu
        self.pv = pv
        self.tu = KnotVector(tu)
        self.tv = KnotVector(tv)
        self.cpts = np.array(c, dtype=flint)
        self.shape = self.cpts[0,0].shape
        self.cpts_array = []
        self.calc_cpts_array(self.cpts, self.cpts_array)

    def calc_cpts_array(self, c: npt.NDArray[flint], arr: list[list[npt.NDArray[flint]]]):
        """Calculate the control points for the derivative spline
        @param c The control points
        @param arr The list to hold the derivative control points
        """
        shape = list(c.shape)
        # First row
        arr.append([c])
        for nv in range(self.pv):
            shape[1] += 1
            r = np.zeros(shape, dtype=flint)
            r[:,:-1] = arr[0][nv][:,:]
            pv = self.pv-nv
            for j in range(shape[1]-1,0,-1):
                dt = self.tv[j+pv]-self.tv[j]
                r[:,j] = 0*r[:,0] if dt == 0 else pv*(r[:,j]-r[:,j-1])/dt
            dt = self.tv[j+pv]-self.tv[j]
            r[:,0] = 0*r[:,0] if dt == 0 else pv*r[:,0]/dt
            arr[0].append(r)
        # Other rows
        for nu in range(1, self.pu+1):
            shape = list(arr[nu-1][0].shape)
            shape[0] += 1
            r = np.zeros(shape, dtype=flint)
            r[:-1] = arr[nu-1][0][:]
            pu = self.pu-(nu-1)
            for i in range(shape[0]-1,0,-1):
                dt = self.tu[i+pu]-self.tu[i]
                r[i] = 0*r[0] if dt == 0 else pu*(r[i]-r[i-1])/dt
            dt = self.tu[pu]-self.tu[0]
            r[0] = 0*r[0] if dt == 0 else pu*r[0]/dt
            arr.append([r])
            for nv in range(self.pv):
                shape[1] += 1
                r = np.zeros(shape, dtype=flint)
                r[:,:-1] = arr[nu][nv][:,:]
                pv = self.pv-nv
                for j in range(shape[1]-1,0,-1):
                    dt = self.tv[j+pv]-self.tv[j]
                    r[:,j] = 0*r[:,0] if dt == 0 else pv*(r[:,j]-r[:,j-1])/dt
                dt = self.tv[j+pv]-self.tv[j]
                r[:,0] = 0*r[:,0] if dt == 0 else pv*r[:,0]/dt
                arr[nu].append(r)

    def _deboor_2d(self, 
                   c: npt.NDArray[flint], 
                   u: Num, 
                   v: Num, 
                   nu: int = 0, 
                   nv: int = 0) -> Point:
        """Perform de Boor's algorithm with arbitrary control points
        @param c The 1-D array of control points
        @param u The u parameter value
        @param v The v parameter value
        @param nu Optional reduction of degree for calculation of u-derivatives
        @param nv Optional reduction of degree for calculation of v-derivatives
        @return The result of de Boor's algorithm
        """
        ku = self.tu.k(u)
        kv = self.tv.k(v)
        pu = self.pu-nu
        pv = self.pv-nv
        c_shape = c.shape
        len_u, len_v = c_shape[:2]
        point_shape = list(c_shape[2:])
        q_shape = [pu+1,pv+1] + point_shape
        q = np.zeros(q_shape, dtype=flint)
        for i in range(pu+1):
            for j in range(pv+1):
                if 0 <= ku-pu+i < len_u and 0 <= kv-pv+j < len_v:
                    q[i,j] = c[ku-pu+i,kv-pv+j]
        for ru in range(pu):
            for i in range(pu,ru,-1):
                l, m = np.clip((i+ku-pu, i+ku-ru), 0, len(self.tu)-1)
                a = (u-self.tu[l])/(self.tu[m]-self.tu[l])
                q[i,:] = a*q[i,:] + (1-a)*q[i-1,:]
        for rv in range(pv):
            for j in range(pv,rv,-1):
                l, m = np.clip((j+kv-pv, j+kv-rv), 0, len(self.tv)-1)
                a = (v-self.tv[l])/(self.tv[m]-self.tv[l])
                q[pu,j] = a*q[pu,j] + (1-a)*q[pu,j-1]                
        return q[pu, pv]

    def d_nv(self, u: Num, v: Num, nu: int, nv: int) -> Point:
        """Evaluate the nu,nv^th order derivative of the spline curve
        @param u The u parameter value
        @param v The v parameter value
        @param nu The order of the u-derivatives
        @param nv The order of the v-derivatives
        @return The value of the derivative curve at the parametric point x
        """
        if nu > self.pu or nv > self.pv:
            return np.zeros(self.shape, dtype=flint)
        else:
            return self._deboor_2d(self.cpts_array[nu][nv], u, v, nu, nv)
