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
from flint import flint

from pyscad.geo.surf import ParaSurf

class MissingMethods(ParaSurf):
    """Conly here to validate errors"""
    pass

class Torus(ParaSurf):
    """A parametrically defined torus using trig functions"""
    shape = (3,)

    def __init__(self, R=3.0, a=1.0):
        self.degenerate_points = {}
        self.R = R
        self.a = a

    def d_nv(self, u, v, nu, nv):
        phi = 2*np.pi*np.array(u, dtype=flint)
        th = 2*np.pi*np.array(v, dtype=flint)
        rho = self.R+self.a*np.cos(th)
        if nu == 0 and nv == 0:
            x = rho*np.cos(phi)
            y = rho*np.sin(phi)
            z = self.a*np.sin(th)
            return np.array([x.T,y.T,z.T]).T
        if nu == 1 and nv == 0:
            x = -rho*np.sin(phi)*2*np.pi
            y = rho*np.cos(phi)*2*np.pi
            z = 0.0*y
            return np.array([x.T,y.T,z.T]).T
        drhodv = -self.a*np.sin(th)*2*np.pi
        if nu == 0 and nv == 1:
            x = drhodv*np.cos(phi)
            y = drhodv*np.sin(phi)
            z = self.a*np.cos(th)*2*np.pi
            return np.array([x.T,y.T,z.T]).T
        if nu == 2 and nv == 0:
            x = -rho*np.cos(phi)*4*np.pi*np.pi
            y = -rho*np.sin(phi)*4*np.pi*np.pi
            z = 0.0*y
            return np.array([x.T,y.T,z.T]).T
        if nu == 1 and nv == 1:
            x = -drhodv*np.sin(phi)*2*np.pi
            y = drhodv*np.cos(phi)*2*np.pi
            z = 0.0*y
            return np.array([x.T,y.T,z.T]).T
        d2rhodv2 = -self.a*np.cos(th)*4*np.pi*np.pi
        if nu == 0 and nv == 2:
            x = d2rhodv2*np.cos(phi)
            y = d2rhodv2*np.sin(phi)
            z = -self.a*np.sin(th)*4*np.pi*np.pi
            return np.array([x.T,y.T,z.T]).T
        raise NotImplementedError("Higher order derivatives are not implmemented")


class TestParaSurf:
    """Validate the behavior of something"""

    def test_exceptions(self):
        mm = MissingMethods()
        with pytest.raises(AttributeError):
            mm(1.0, 1.0)
        mm.shape = (1,)
        with pytest.raises(NotImplementedError):
            mm.d(1.0, 1.0, 1, 0)
        with pytest.raises(NotImplementedError):
            mm.d(1.0, 1.0, 0, 1)

    def test_normal_scalar(self):
        t = Torus()
        assert np.alltrue( t.norm(0,0) == [1, 0, 0] )

    def test_mean_curvature_scalar(self):
        t = Torus()
        assert t.k_mean(0,0) == -0.625

    def test_gaussian_curvature_scalar(self):
        t = Torus()
        assert t.k_gauss(0,0) == 0.25

    def test_principal_curvatures_scalar(self):
        t = Torus()
        kp, km = t.k_princ(0,0)
        assert kp == -0.25
        assert km == -1

    def test_degenerate_point(self):
        t = Torus(R=0, a=1)
        assert np.alltrue( t(0,0) == [1,0,0] )
        assert np.alltrue( t(0.25,0) == [0,1,0] )
        assert np.alltrue( t(0,0.25) == [0,0,1] )
        assert np.alltrue( t(0.25,0.25) == [0,0,1] )
        assert np.alltrue( t.norm(0,0.25) == [0,0,1] )