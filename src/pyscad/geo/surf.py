## @file surf.py 
"""\
Space surface base class
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

from ..types import *

class ParaSurf:
    """A parametric surface from u,v to R^3"""

    def __init__(self):
        self.degenerate_points = {}

    def __call__(self, u: Num, v: Num) -> Point:
        """Evaluate the surface
        @param u The u parametric value
        @param v The v parametric value
        @return The value of the surface
        """
        return self.d(u, v, 0, 0)

    @staticmethod
    def to_array(u: npt.ArrayLike, v: npt.ArrayLike, dtype: npt.DTypeLike = flint) -> tuple[npt.NDArray[flint], npt.NDArray[flint]]:
        if np.shape(u) != np.shape(v):
            raise ValueError("Shape of the u and v parameter arguments must match")
        return np.array(u, dtype=dtype), np.array(v, dtype=dtype)

    def d(self, u: Num, v: Num, nu: int, nv: int) -> Point:
        """Evaluate the nu^th,nv^th partial derivative of the surface
        @param u The u parametric value
        @param v The v parametric value
        @param nu The order of the u partial derivative
        @param nv The order of the v partial derivative
        @return The nu^th, nv^th partial derivative
        """
        u, v = self.to_array(u, v)
        out_shape = list(np.shape(nu)) + list(np.shape(u)) + list(self.shape)
        out_array = np.zeros(out_shape, dtype=flint)
        pre = [np.s_[:]]*len(np.shape(nu))
        post = [np.s_[:]]*len(self.shape)
        for idx in np.ndindex(*np.shape(u)):
            array_idx = tuple(pre + list(idx) + post)
            out_array[array_idx] = self.d_vec(u[idx], v[idx], nu, nv)
        return out_array

    def d_vec(self, u: Num, v: Num, nu: int, nv: int) -> Point:
        """Evaluate the partial derivative vectorized over the derivative orders
        @param u The u parametric value
        @param v The v parametric value
        @param nu The order of the u partial derivative
        @param nv The order of the v partial derivative
        @return The nu^th, nv^th partial derivative
        """
        nu, nv = self.to_array(nu, nv, dtype=int)
        out_shape = list(np.shape(nu)) + list(self.shape)
        out_array = np.zeros(out_shape, dtype=flint)
        for idx in np.ndindex(*np.shape(nu)):
            out_array[idx] = self.d_nv(u, v, nu[idx], nv[idx])
        return out_array

    def d_nv(self, u: Num, v: Num, nu: int, nv: int) -> Point:
        """Evaluate the partial derivative non-vectorized over all arguments
        @param u The u parametric value
        @param v The v parametric value
        @param nu The order of the u partial derivative
        @param nv The order of the v partial derivative
        @return The nu^th, nv^th partial derivative
        """
        raise NotImplementedError(("Virtual Method, must redefine"))

    def norm(self, u: Num, v: Num) -> Vec:
        """Calculate the normal vector of the surface
        @param u The u parameter
        @param v The v parameter
        @return The normal vector of the surface at (u,v)
        """
        u, v = self.to_array(u, v)
        out_shape = list(np.shape(u)) + list(self.shape)
        out_array = np.zeros(out_shape, dtype=flint)
        for idx in np.ndindex(*np.shape(u)):
            du, dv = self.d_vec(u[idx],v[idx],[1,0],[0,1])
            if mag(du) == 0:
                if mag(dv) == 0:
                    out_array[idx] = np.zeros(self.shape, dtype=flint)
                else:
                    out_array[idx] = self.norm_zero_len(u[idx],v[idx],dv,'u')
            elif mag(dv) == 0:
                out_array[idx] = self.norm_zero_len(u[idx],v[idx],du,'v')
            else:
                n = np.cross(du, dv)
                m = mag(n)
                if mag(n) == 0:
                    out_array[idx] = np.zeros(self.shape, dtype=flint)
                else:
                    out_array[idx] = n/m
        return out_array

    def norm_zero_len(self, u: Num, v: Num, d: Vec, zdir: str) -> Vec:
        """Calculate the normal vector for a degenerate point on a zero length edge
        @param u The u parameter
        @param v The v parameter
        @param d The non-zero partial derivative
        @param zdir The zero parametric direction
        """
        # Memorization to avoid recalculating
        for ku,kv in self.degenerate_points.keys():
            if u == ku and v == kv:
                return self.degenerate_points((ku, kv))
        step = 0.1
        nu, nv =  (0, 1) if zdir == 'u' else (1, 0)
        p0 = self.d_vec(u, v, 0, 0)
        p1 = self.d_vec(u+2*step*nv, v+2*step*nu, 0, 0)
        if np.alltrue(p0 == p1):
            d1 = self.d_vec(u+step*nv, v+step*nu, nu, nv)
            n1 = np.cross(d, d1)
            m1 = mag(n1)
            d2 = self.d_vec(u+2*step*nv, v+2*step*nu, nu, nv)
            n2 = np.cross(d,d2)
            m2 = mag(n2)
            if m1 != 0 and m2 != 0 and np.alltrue(n1/m1 == n2/m2):
                self.degenerate_points[(u,v)] = n2/m2
                return n2/m2
        out = np.zeros(self.shape, dtype=flint)
        self.degenerate_points[(u,v)] = out
        return out

    def ff_nv(self, u: Num, v: Num) -> tuple[Num, Num, Num, Num, Num, Num]:
        """Calculate the components of the first and second fundamental form non-vectorized
        @param u The u parameter
        @param v The v parameter
        @return The E, F, G, L, M, and N components of the second fundamental form
        """
        du, dv, duu, duv, dvv = self.d_vec(u,v,[1,0,2,1,0],[0,1,0,1,2])
        n = np.cross(du, dv)
        m = mag(n)
        n = n if m == 0 else n/m
        E = du.dot(du)
        F = du.dot(dv)
        G = dv.dot(dv)
        L = n.dot(duu)
        M = n.dot(duv)
        N = n.dot(dvv)
        return E,F,G,L,M,N

    def k_mean(self, u: Num, v: Num) -> Num:
        """Calculate the mean curvature of the surface
        @param u The u parameter
        @param v The v parameter
        @return The mean curvature of the surface at (u,v)
        """
        u, v = self.to_array(u, v)
        out_array = np.zeros(np.shape(u), dtype=flint)
        for idx in np.ndindex(*np.shape(u)):
            E, F, G, L, M, N = self.ff_nv(u[idx], v[idx])
            out_array[idx] = (E*N-2*F*M+G*L)/(2*(E*G-F*F))
        return out_array
    
    def k_gauss(self, u: Num, v: Num) -> Num:
        """Calculate the Gaussian curvature of the surface
        @param u The u parameter
        @param v The v parameter
        @return The mean curvature of the surface at (u,v)
        """
        u, v = self.to_array(u, v)
        out_array = np.zeros(np.shape(u), dtype=flint)
        for idx in np.ndindex(*np.shape(u)):
            E, F, G, L, M, N = self.ff_nv(u[idx], v[idx])
            out_array[idx] =  (L*N-M*M)/(E*G-F*F)
        return out_array

    def k_princ(self, u: Num, v: Num) -> tuple[Num, Num]:
        """Calculate the principle curvatures of the surface
        @param u The u parameter
        @param v The v parameter
        @return The k+ and k- principal curvature
        """
        u, v = self.to_array(u, v)
        out_shape = tuple([2] + list(np.shape(u)))
        out_array = np.zeros(out_shape, dtype=flint)
        for idx in np.ndindex(*np.shape(u)):
            E, F, G, L, M, N = self.ff_nv(u[idx], v[idx])
            a = (E*G - F*F)
            b = (L*G - 2*M*F + N*E)/(2*a)
            c =(L*N - M*M)/a
            d = np.sqrt(b*b - c)
            kp_idx = tuple([0] + list(idx))
            out_array[kp_idx] = b + d
            km_idx = tuple([1] + list(idx))
            out_array[km_idx] = b - d
        return out_array


class Plane(ParaSurf):

    def __init__(self, p0, pu, pv):
        self.cpts = np.array([p0, pu, pv], dtype=flint)
        self.shape = self.cpts[0].shape
        du = self.cpts[1]-self.cpts[0]
        dv = self.cpts[2]-self.cpts[0]
        if mag(np.cross(du,dv)) == 0:
            raise ValueError("Points defining the plane cannot be co-linear")

    def d_nv(self, u: Num, v: Num, nu: int, nv: int) -> Point:
        p0 = self.cpts[0]
        du = self.cpts[1]-p0
        dv = self.cpts[2]-p0
        if nu == 0 and nv == 0:
            return p0 + du*u + dv*v
        elif nu == 1 and nv == 0:
            return du
        elif nu == 0 and nv == 1:
            return dv
        else:
            return np.zeros(self.shape, dtype=flint)
