import unittest

import numpy as np

from ..flint import v_flint
from ..cpoint import cp_vectorize
from ..surf import *

class MissingMethods(ParaSurf):
    """Only hear to validate errors"""
    pass


class Torus(ParaSurf):
    """A parametrically defined torus using trig functions"""

    R = 2.0
    a = 1.0

    @cp_vectorize
    def __call__(self, u, v):
        """Evaluate the surface"""
        phi = u/2*np.pi
        th = v/2*np.pi
        rho = self.R+self.a*np.cos(th)
        x = rho*np.cos(phi)
        y = rho*np.sin(phi)
        z = self.a*np.sin(th)
        return v_flint([x,y,z])

    @cp_vectorize(ignore=(2,3))
    def d(self, u, v, nu, nv):
        """Evaluate first or second derivatives"""
        if nu == 0 and nv == 0:
            return self.__call__(u,v)
        phi = u/2*np.pi
        th = v/2*np.pi
        rho = self.R+self.a*np.cos(th)
        if nu == 1 and nv == 0:
            x = -rho*np.sin(phi)/2*np.pi
            y = rho*np.cos(phi)/2*np.pi
            z = 0.0
            return v_flint([x,y,z])
        drhodv = -self.a*np.sin(th)/2*np.pi
        if nu == 0 and nv == 1:
            x = drhodv*np.cos(phi)
            y = drhodv*np.sin(phi)
            z = self.a*np.cos(th)/2*np.pi
            return v_flint([x,y,z])
        if nu == 2 and nv == 0:
            x = -rho*np.cos(phi)/4*np.pi*np.pi
            y = -rho*np.sin(phi)/4*np.pi*np.pi
            z = 0.0
            return v_flint([x,y,z])
        if nu == 1 and nv == 1:
            x = -drhodv*np.sin(phi)/2*np.pi
            y = drhodv*np.cos(phi)/2*np.pi
            z = 0.0
            return v_flint([x,y,z])
        d2rhodv2 = -self.a*np.cos(th)/4*np.pi*np.pi
        if nu == 0 and nv == 2:
            x = d2rhodv2*np.cos(phi)
            y = d2rhodv2*np.sin(phi)
            z = -self.a*np.sin(th)/4*np.pi*np.pi
            return v_flint([x,y,z])


class TestSurfProp(unittest.TestCase):
    """Test suite for general properties of parametric surfaces"""

    def setUp(self):
        self.torus = Torus()

    def test_error(self):
        """Validate that exception is thrown if methods aren't yet defined"""
        mm = MissingMethods()
        with self.assertRaises(NotImplementedError):
            mm(1.0, 1.0)

    def test_d_tri(self):
        """Validate generating triangular array of derivatives"""
        res = self.torus.d_tri(0,0,2)
        self.assertEqual(len(res), 3)
        self.assertEqual(len(res[0]), 3)
        self.assertEqual(len(res[1]), 2)
        self.assertEqual(len(res[2]), 1)
        for i in range(3):
            for j in range(3-i):
                comp = self.torus.d(0,0,i,j)
                self.assertTrue(np.alltrue(res[i][j]==comp))

    def test_d_rect(self):
        """Validate generating rectangular array of derivatives"""
        res = self.torus.d_rect(0,0,1,1)
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 2)
        self.assertEqual(len(res[1]), 2)
        for i in range(2):
            for j in range(2):
                comp = self.torus.d(0,0,i,j)
                self.assertTrue(np.alltrue(res[i][j]==comp))

    def test_normal(self):
        """Validate the normal vector for the torus"""
        uvpts = [[0,0],[1,0],[0,1],[1,1]]
        tnorms = v_flint([
                    [1,0,0],
                    [0,1,0],
                    [0,0,1],
                    [0,0,1],
                 ])
        for uv, tn  in zip(uvpts, tnorms):
            n = self.torus.normal(*uv)
            self.assertEqual(np.dot(n, tn), 1.0)

    def test_normal_ufunc(self):
        """Validate the normal vector for multiple points on the torus"""
        raise NotImplemented

    def test_k_mean(self):
        """Validate the mean curvature for the torus"""
        raise NotImplemented

    def test_k_gaussian(self):
        """Validate the Gaussian curvature for the torus"""
        raise NotImplemented
    
    def test_k_principal(self):
        """Validate the principle curvatures"""
        raise NotImplemented
    
    def test_k_princ_vec(self):
        """Validate the principle curvatures and eigenvectors"""
        raise NotImplemented