import unittest

import numpy as np

from ..flint import v_flint
from ..curves import SpaceCurve
from ..bspline import BSpline
from .test_curves import simple_basis, simple_basis_d1, simple_basis_d2

class TestInit(unittest.TestCase):
    """Test the internal structure and class interface for a b-spline"""

    def test_init(self):
        """Validate the initialization works correctly"""
        p = 2
        t = [0,1,2,3]
        self.bs = BSpline([[1,-1]],p,t)
        self.assertIsInstance(self.bs, SpaceCurve)
        self.assertIsInstance(self.bs, BSpline)
        self.assertEqual(self.bs.p, 2)
        self.assertIsInstance(self.bs.c, np.ndarray)
        with self.assertRaises(ValueError):
            bad = BSpline(self.bs.c, self.bs.p, [0,1])


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


class TestEval(unittest.TestCase):
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

class TestDerivative(unittest.TestCase):
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
