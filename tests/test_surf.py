## @file test_surf.py
"""\
Validate test for parametric surface base class
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

import pytest

import numpy as np
import flint as flint

from pyscad.geo.surf import ParaSurf

class MissingMethods(ParaSurf):
    """Conly here to validate errors"""
    pass

class Torus(ParaSurf):
    """A parametrically defined torus using trig functions"""

    R = 3.0
    a = 1.0

    def __call__(self, u, v):
        return self.d(u,v,0,0)

    def d(self, u, v, nu, nv):
        if nu == 0 and nv == 0:
            return self.__call__(u,v)
        phi = u/2*np.pi
        th = v/2*np.pi
        rho = self.R+self.a*np.cos(th)
        if nu == 0 and nv == 0:
            x = rho*np.cos(phi)
            y = rho*np.sin(phi)
            z = self.a*np.sin(th)
            return np.array([x.T,y.T,z.T]).T
        if nu == 1 and nv == 0:
            x = -rho*np.sin(phi)/2*np.pi
            y = rho*np.cos(phi)/2*np.pi
            z = 0.0*y
            return np.array([x.T,y.T,z.T]).T
        drhodv = -self.a*np.sin(th)/2*np.pi
        if nu == 0 and nv == 1:
            x = drhodv*np.cos(phi)
            y = drhodv*np.sin(phi)
            z = self.a*np.cos(th)/2*np.pi
            return np.array([x.T,y.T,z.T]).T
        if nu == 2 and nv == 0:
            x = -rho*np.cos(phi)/4*np.pi*np.pi
            y = -rho*np.sin(phi)/4*np.pi*np.pi
            z = 0.0*y
            return np.array([x.T,y.T,z.T]).T
        if nu == 1 and nv == 1:
            x = -drhodv*np.sin(phi)/2*np.pi
            y = drhodv*np.cos(phi)/2*np.pi
            z = 0.0*y
            return np.array([x.T,y.T,z.T]).T
        d2rhodv2 = -self.a*np.cos(th)/4*np.pi*np.pi
        if nu == 0 and nv == 2:
            x = d2rhodv2*np.cos(phi)
            y = d2rhodv2*np.sin(phi)
            z = -self.a*np.sin(th)/4*np.pi*np.pi
            return np.array([x.T,y.T,z.T]).T


class TestParaSurf:
    """Vlaidate the behavior of something"""

    def test_exceptions(self):
        mm = MissingMethods()
        with pytest.raises(NotImplementedError):
            mm(1.0, 1.0)
        with pytest.raises(NotImplementedError):
            mm.d(1.0, 1.0, 1, 0)
        with pytest.raises(NotImplementedError):
            mm.d(1.0, 1.0, 0, 1)

