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

    def __call__(self, u: Num, v: Num) -> Point:
        raise NotImplementedError("Virtual method, must redefine")

    def d(self, u: Num, v: Num, nu: int, nv: int) -> Point:
        raise NotImplementedError("Virtual method, must redefine")

    def norm(self, u: Num, v: Num) -> Vec:
        """Calculate the normal vector of the surface
        @param u The u parameter
        @param v The v parameter
        @return The normal vector of the surface at (u,v)
        """
        du = self.d(u,v,1,0)
        dv = self.d(u,v,0,1)
        n = np.cross(du, dv)
        return n/mag(n)[...,np.newaxis]

    def fff(self, u: Num, v: Num) -> tuple[Num, Num, Num]:
        """Calculate the components of the first fundamental form
        @param u The u parameter
        @param v The v parameter
        @return The E, F, and G components of the first fundamental form
        """
        du = self.d(u,v,1,0)
        dv = self.d(u,v,0,1)
        E = (du*du).sum(axis=-1)
        F = (du*dv).sum(axis=-1)
        G = (dv*dv).sum(axis=-1)
        return E, F, G

    def sff(self, u: Num, v: Num) -> tuple[Num, Num, Num]:
        """Calculate the components of the second fundamental form
        @param u The u parameter
        @param v The v parameter
        @return The L, M, and N components of the second fundamental form
        """
        n = self.norm(u,v)
        duu = self.d(u,v,2,0)
        duv = self.d(u,v,1,1)
        dvv = self.d(u,v,0,2)
        L = (n*duu).sum(axis=-1)
        M = (n*duv).sum(axis=-1)
        N = (n*dvv).sum(axis=-1)
        return L, M, N

    def k_mean(self, u: Num, v: Num) -> Num:
        """Calculate the mean curvature of the surface
        @param u The u parameter
        @param v The v parameter
        @return The mean curvature of the surface at (u,v)
        """
        E, F, G = self.fff(u,v)
        L, M, N = self.sff(u,v)
        return (E*N-2*F*M+G*L)/(2*(E*G-F*F))
    
    def k_gauss(self, u: Num, v: Num) -> Num:
        """Calculate the Gaussian curvature of the surface
        @param u The u parameter
        @param v The v parameter
        @return The mean curvature of the surface at (u,v)
        """
        E, F, G = self.fff(u,v)
        L, M, N = self.sff(u,v)
        return (L*N-M*M)/(E*G-F*F)

    def k_princ(self, u: Num, v: Num) -> tuple[Num, Num]:
        """Calculate the principle curvatures of the surface
        @param u The u parameter
        @param v The v parameter
        @return The k+ and k- principal curvature
        """
        E, F, G = self.fff(u,v)
        L, M, N = self.sff(u,v)
        a = (E*G - F*F)
        b = (L*G - 2*M*F + N*E)/(2*a)
        c =(L*N - M*M)/a
        d = np.sqrt(b*b - c)
        kp = b + d
        km = b - d
        return kp, km
