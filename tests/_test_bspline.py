import unittest

import numpy as np

from ..flint import v_flint
from ..curves import ParaCurve
from ..surf import ParaSurf
from ..bspline import BSpline, BSplineSurf
from .test_curves import simple_basis, simple_basis_d1, simple_basis_d2

class TestInit(unittest.TestCase):
    """Test the internal structure and class interface for a b-spline"""

    def test_curve(self):
        """Validate the initialization works correctly"""
        p = 2
        t = [0,1,2,3]
        self.bs = BSpline([[1,-1]],p,t)
        self.assertIsInstance(self.bs, ParaCurve)
        self.assertIsInstance(self.bs, BSpline)
        self.assertEqual(self.bs.p, 2)
        self.assertIsInstance(self.bs.c, np.ndarray)
        with self.assertRaises(ValueError):
            bad = BSpline(self.bs.c, self.bs.p, [0,1])

    def test_surf(self):
        """Validate the surface initialization works correctly"""
        p = 2
        t = [0,1,2,3]
        c = [[[1,1,1]]]
        s = BSplineSurf(c, p, p, t, t)
        self.assertIsInstance(s, ParaSurf)
        self.assertIsInstance(s, BSplineSurf)
        self.assertEqual(s.pu, p)
        self.assertEqual(s.pv, p)
        self.assertIsInstance(s.c, np.ndarray)
        self.assertEqual(s.c.shape, (1,1,3))
        with self.assertRaises(ValueError):
            bad = BSplineSurf([[[1,1,1]]],3,p,t,t)
        with self.assertRaises(ValueError):
            bad = BSplineSurf([[[1,1,1]]],p,p,t,[1,2,3])


class TestCPointTypes(unittest.TestCase):
    """Test evaluating for different control point sizes"""

    def test_call_flint(self):
        """Validate the behavior for scalar control points"""
        p = 2
        t = [0,1,2,3]
        self.bsf = BSpline([1.0],p,t)
        for tval in np.linspace(0,3,10):
            a = self.bsf(tval)
            b = simple_basis(tval)
            # value were close, but not close enough, needed to grow
            a._grow()
            self.assertEqual(a, b)
    
    def test_call_flint_surf(self):
        """Validate the behavior for scalar control points"""
        p = 2
        t = [0,1,2,3]
        s = BSplineSurf([[1.0]],p,p,t,t)
        uu = np.linspace(0,3,10)
        vv = np.linspace(0,3,10)
        U, V = np.meshgrid(uu,vv)
        U = U.reshape((100,))
        V = V.reshape((100,))
        for u,v in zip(U,V):
            a = s(u,v)
            b = simple_basis(u)*simple_basis(v)
            # value were close, but not close enough, needed to grow
            a._grow()
            self.assertEqual(a, b)

    def test_call_1D(self):
        """Validate the behavior for 1-D control points"""
        p = 2
        t = [0,1,2,3]
        self.bs1 = BSpline([[1]],p,t)
        for tval in np.linspace(0,3,10):
            a = self.bs1(tval)
            b = simple_basis(tval)
            a[0]._grow()
            self.assertEqual(a[0], b)

    def test_call_2D(self):
        """Validate the behavior for 2-D control points"""
        p = 2
        t = [0,1,2,3]
        self.bs = BSpline([[1,-1]],p,t)
        for tval in np.linspace(0,3,10):
            a = self.bs(tval)
            b = simple_basis(tval)
            a[0]._grow()
            a[1]._grow()
            self.assertEqual(a[0], b)
            self.assertEqual(a[1], -b)


class TestCurveEval(unittest.TestCase):
    """Test the evaluating the b-spline structure"""

    def setUp(self):
        """We will be testing against simple basis on 0123 knot vector"""
        p = 2
        t = [0,1,2,3]
        self.bs = BSpline([[1,-1]],p,t)

    def test_call_scalar(self):
        """Validate calling with a single parameter value"""
        for tval in np.linspace(0,3,10):
            a = self.bs(tval)
            b = simple_basis(tval)
            a[0]._grow()
            a[1]._grow()
            self.assertEqual(a[0], b)
            self.assertEqual(a[1], -b)

    def test_call_vec(self):
        """Validate calling for 1-D vector of parameter value"""
        t = np.linspace(0,3,10)
        a = self.bs(t)
        b = np.empty((10,2), dtype=object)
        for i, tval in enumerate(t):
            b[i] = v_flint([simple_basis(tval), -simple_basis(tval)])
            a[i,0]._grow()
            a[i,1]._grow()
        self.assertTrue(np.alltrue(a==b))

    def test_call_array(self):
        """Validate calling for 2-D array of parameter value"""
        t = np.array([np.linspace(i,i+1,10) for i in range(3)]).T
        a = self.bs(t)
        b = np.empty((10,3,2), dtype=object)
        for i in range(10):
            for j in range(3):
                b[i,j] = v_flint([simple_basis(t[i,j]), -simple_basis(t[i,j])])
                a[i,j,0]._grow()
                a[i,j,1]._grow()
        self.assertTrue(np.alltrue(a==b))


class TestSurfEval(unittest.TestCase):
    """Test the evaluating the b-spline structure"""

    def setUp(self):
        """We will be testing against simple basis on 0123 knot vector"""
        p = 2
        t = [0,1,2,3]
        self.s = BSplineSurf([[[1,-1]]], p, p, t, t)

    def test_call_scalar(self):
        uu = np.linspace(0,3,10)
        vv = np.linspace(0,3,10)
        U, V = np.meshgrid(uu,vv)
        U = U.reshape((100,))
        V = V.reshape((100,))
        for u,v in zip(U,V):
            a = self.s(u,v)
            bval = simple_basis(u)*simple_basis(v)
            b = v_flint([bval, -bval])
            # value were close, but not close enough, needed to grow
            a[0]._grow()
            a[1]._grow()
            self.assertTrue(np.alltrue(a==b))

    def test_call_vector(self):
        uu = np.linspace(0,3,10)
        vv = np.full_like(uu, 1.5)
        a = self.s(uu,vv)
        b = np.empty((10,2), dtype=object)
        for i, pvals in enumerate(zip(uu,vv)):
            u,v = pvals
            bval = simple_basis(u)*simple_basis(v)
            b[i] = v_flint([bval, -bval])
            # value were close, but not close enough, needed to grow
            a[i,0]._grow()
            a[i,1]._grow()
        self.assertTrue(np.alltrue(a==b))

    def test_call_array(self):
        uu = np.linspace(0,3,10)
        vv = np.linspace(0,3,10)
        U, V = np.meshgrid(uu,vv)
        a = self.s(U, V)
        b = np.empty((10,10,2), dtype=object)
        for i in range(10):
            for j in range(10):
                u, v = U[i,j], V[i,j]
                bval = simple_basis(u)*simple_basis(v)
                b[i,j] = v_flint([bval, -bval])
                # value were close, but not close enough, needed to grow
                a[i,j,0]._grow()
                a[i,j,1]._grow()
        self.assertTrue(np.alltrue(a==b))


class TestCurveDerivative(unittest.TestCase):
    """Test the evaluating the derivative of the b-spline"""

    def setUp(self):
        """We will be testing against simple basis on 0123 knot vector"""
        p = 2
        t = [0,1,2,3]
        self.bs = BSpline([[1,-1]],p,t)

    def test_d1_scalar(self):
        """Validate first derivative with a single parameter value"""
        for tval in np.linspace(0,3,10):
            a = self.bs.d(tval)
            b = simple_basis_d1(tval)
            a[0]._grow()
            a[1]._grow()
            self.assertEqual(a[0], b)
            self.assertEqual(a[1], -b)

    def test_d1_vec(self):
        """Validate first derivative for 1-D vector of parameter value"""
        t = np.linspace(0,3,10)
        a = self.bs.d(t)
        b = np.empty((10,2), dtype=object)
        for i, tval in enumerate(t):
            b[i] = v_flint([simple_basis_d1(tval), -simple_basis_d1(tval)])
            a[i,0]._grow()
            a[i,1]._grow()
        self.assertTrue(np.alltrue(a==b))

    def test_d2_vec(self):
        """Validate second derivative for 1-D vector of parameter value"""
        t = np.linspace(0,3,10)
        a = self.bs.d(t, 2)
        b = np.empty((10,2), dtype=object)
        for i, tval in enumerate(t):
            b[i] = v_flint([simple_basis_d2(tval), -simple_basis_d2(tval)])
            a[i,0]._grow()
            a[i,1]._grow()
        self.assertTrue(np.alltrue(a==b))

    def test_dlist_vec(self):
        """Validate generating a list of value, first and second derivative"""
        t = np.linspace(0,3,10)
        av, ad1, ad2 = self.bs.d_list(t, 2)
        bv = np.empty((10,2), dtype=object)
        bd1 = np.empty((10,2), dtype=object)
        bd2 = np.empty((10,2), dtype=object)
        for i, tval in enumerate(t):
            bv[i] = v_flint([simple_basis(tval), -simple_basis(tval)])
            bd1[i] = v_flint([simple_basis_d1(tval), -simple_basis_d1(tval)])
            bd2[i] = v_flint([simple_basis_d2(tval), -simple_basis_d2(tval)])
            for a in [av, ad1, ad2]:
                a[i,0]._grow()
                a[i,1]._grow()
        self.assertTrue(np.alltrue(av==bv))
        self.assertTrue(np.alltrue(ad1==bd1))
        self.assertTrue(np.alltrue(ad2==bd2))

    def test_d1_array(self):
        """Validate calling for 2-D array of parameter value"""
        t = np.array([np.linspace(i,i+1,10) for i in range(3)]).T
        a = self.bs.d(t)
        b = np.empty((10,3,2), dtype=object)
        for i in range(10):
            for j in range(3):
                b[i,j] = v_flint([simple_basis_d1(t[i,j]), -simple_basis_d1(t[i,j])])
                a[i,j,0]._grow()
                a[i,j,1]._grow()
        self.assertTrue(np.alltrue(a==b))

class TestSurfDerivatives(unittest.TestCase):
    """Test suite for derivatives of the b-spline surface"""

    def setUp(self):
        """We will be testing against simple basis on 0123 knot vector"""
        p = 2
        t = [0,1,2,3]
        self.s = BSplineSurf([[[1]]], p, p, t, t)

    def test_du_scalar(self):
        """Validate a single 'u' derivative for scalar input"""
        for u in np.linspace(0,3,10):
            a = self.s.d(u,1.5,1,0)
            b = v_flint([simple_basis_d1(u)*simple_basis(1.5)])
            a[0]._grow()
            self.assertEqual(a, b)
 
    def test_du_vector(self):
        """Validate a single 'u' derivative for vector input"""
        u = np.linspace(0,3,10)
        v = np.full_like(u, 1.5)
        a = self.s.d(u,v,1,0)
        b = np.empty(a.shape, dtype=a.dtype)
        for i in range(10):
            b[i] = v_flint([simple_basis_d1(u[i])*simple_basis(1.5)])
            a[i,0]._grow()
        self.assertTrue(np.alltrue(a==b))

    def test_du_array(self):
        """Validate a single 'u' derivative for array input"""
        u = np.linspace(0,3,10)
        U, V = np.meshgrid(u,u)
        a = self.s.d(U,V,1,0)
        b = np.empty(a.shape, dtype=a.dtype)
        for i in range(10):
            for j in range(10):
                b[i,j] = v_flint([simple_basis_d1(U[i,j])*simple_basis(V[i,j])])
                a[i,j,0]._grow()
        self.assertTrue(np.alltrue(a==b))

    def test_dv_scalar(self):
        """Validate a single 'v' derivative for scalar input"""
        for v in np.linspace(0,3,10):
            a = self.s.d(1.5,v,0,1)
            b = v_flint([simple_basis_d1(v)*simple_basis(1.5)])
            a[0]._grow()
            self.assertEqual(a, b)
 
    def test_dv_vector(self):
        """Validate a single 'v' derivative for vector input"""
        v = np.linspace(0,3,10)
        u = np.full_like(v, 1.5)
        a = self.s.d(u,v,0,1)
        b = np.empty(a.shape, dtype=a.dtype)
        for i in range(10):
            b[i] = v_flint([simple_basis_d1(v[i])*simple_basis(1.5)])
            a[i,0]._grow()
        self.assertTrue(np.alltrue(a==b))

    def test_dv_array(self):
        """Validate a single 'v' derivative for array input"""
        u = np.linspace(0,3,10)
        U, V = np.meshgrid(u,u)
        a = self.s.d(U,V,0,1)
        b = np.empty(a.shape, dtype=a.dtype)
        for i in range(10):
            for j in range(10):
                b[i,j] = v_flint([simple_basis_d1(V[i,j])*simple_basis(U[i,j])])
                a[i,j,0]._grow()
        self.assertTrue(np.alltrue(a==b))

    def test_dudv_array(self):
        """Validate combined 'u' and 'v' derivatives over the surface"""
        u = np.linspace(0,3,10)
        U, V = np.meshgrid(u,u)
        a = self.s.d(U,V,1,1)
        b = np.empty(a.shape, dtype=a.dtype)
        for i in range(10):
            for j in range(10):
                b[i,j] = v_flint([simple_basis_d1(V[i,j])*simple_basis_d1(U[i,j])])
                a[i,j,0]._grow()
        self.assertTrue(np.alltrue(a==b))

    def test_d_rect_array(self):
        """Validate generating triangular array of partial derivatives"""
        u = np.linspace(0,3,10)
        U, V = np.meshgrid(u,u)
        aa = self.s.d_rect(U,V,1,1)
        for ii in range(2):
            for jj in range(2):
                a = aa[ii][jj]
                b = np.empty(a.shape, dtype=a.dtype)
                for i in range(10):
                    for j in range(10):
                        if ii == 0 and jj == 0:
                            b[i,j] = v_flint([simple_basis(V[i,j])*simple_basis(U[i,j])])
                        elif ii == 0 and jj == 1:
                            b[i,j] = v_flint([simple_basis_d1(V[i,j])*simple_basis(U[i,j])])
                        elif ii == 1 and jj == 0:
                            b[i,j] = v_flint([simple_basis(V[i,j])*simple_basis_d1(U[i,j])])
                        elif ii == 1 and jj == 1:
                            b[i,j] = v_flint([simple_basis_d1(V[i,j])*simple_basis_d1(U[i,j])])
                        a[i,j,0]._grow()
                self.assertTrue(np.alltrue(a==b))

    def test_d_tri_array(self):
        """Validate generating triangular array of partial derivatives"""
        u = np.linspace(0,3,10)
        U, V = np.meshgrid(u,u)
        aa = self.s.d_tri(U,V,2)
        for ii in range(3):
            for jj in range(3-ii):
                a = aa[ii][jj]
                b = np.empty(a.shape, dtype=a.dtype)
                for i in range(10):
                    for j in range(10):
                        if ii == 0 and jj == 0:
                            b[i,j] = v_flint([simple_basis(V[i,j])*simple_basis(U[i,j])])
                        elif ii == 0 and jj == 1:
                            b[i,j] = v_flint([simple_basis_d1(V[i,j])*simple_basis(U[i,j])])
                        elif ii == 0 and jj == 2:
                            b[i,j] = v_flint([simple_basis_d2(V[i,j])*simple_basis(U[i,j])])
                        elif ii == 1 and jj == 0:
                            b[i,j] = v_flint([simple_basis(V[i,j])*simple_basis_d1(U[i,j])])
                        elif ii == 1 and jj == 1:
                            b[i,j] = v_flint([simple_basis_d1(V[i,j])*simple_basis_d1(U[i,j])])
                        elif ii == 2 and jj == 0:
                            b[i,j] = v_flint([simple_basis(V[i,j])*simple_basis_d2(U[i,j])])
                        a[i,j,0]._grow()
                self.assertTrue(np.alltrue(a==b))
