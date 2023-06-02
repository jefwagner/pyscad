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

from pyscad.geo.surf import ParaSurf, Plane

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


class TestPlane:

    def test_init(self):
        p = Plane([0,0,0],[1,0,1],[0,1,1])
        assert isinstance(p, ParaSurf)
        assert isinstance(p, Plane)
        assert p.shape == (3,)
    
    def test_exceptions(self):
        with pytest.raises(ValueError):
            p = Plane([0,0,0],[0,0,0],[1,1,1])
        with pytest.raises(ValueError):
            p = Plane([0,0,0],[1,1,1],[1,1,1])
        with pytest.raises(ValueError):
            p = Plane([0,0,0],[1,1,1],[2,2,2])

    def test_eval(self):
        p0 = [0,0,0]
        p1 = [1,0,1]
        p2 = [0,1,1]
        p3 = [1,1,2]
        p = Plane(p0, p1, p2)
        U, V = np.meshgrid([0,1],[0,1])
        pts = p(U,V)
        assert np.alltrue( pts[0,0] == p0 )
        assert np.alltrue( pts[0,1] == p1 )
        assert np.alltrue( pts[1,0] == p2 )
        assert np.alltrue( pts[1,1] == p3 )
        assert np.alltrue( p(0.5, 0.5) == [0.5, 0.5, 1] )

    def test_deriv(self):
        p0 = [0,0,0]
        p1 = [1,0,1]
        p2 = [0,1,1]
        p = Plane(p0, p1, p2)
        u = np.linspace(0,1,10)
        v = np.linspace(1,0,10)
        U, V = np.meshgrid(u, v)
        du = p.d(U,V,1,0)
        for idx in np.ndindex((10,10)):
            assert np.alltrue( du[idx] == p1 )
        dv = p.d(U,V,0,1)
        for idx in np.ndindex((10,10)):
            assert np.alltrue( dv[idx] == p2 )
        for nu in range(1,3):
            for nv in range(1,3):
                dd = p.d(U,V,nu,nv)
                for idx in np.ndindex((10,10)):
                    assert np.alltrue( dd[idx] == [0,0,0] )
