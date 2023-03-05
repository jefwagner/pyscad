import unittest

import numpy as np

from ..flint import flint, v_flint
from ..cpoint import *

class TestCpConversions(unittest.TestCase):
    """Test for converting between arrays of CPoints and components"""

    def test_conv_0D(self):
        pts = np.array([1.0, 2.0])
        x = 1.0
        y = 2.0
        # Confirm the pts->components works
        xx,yy = to_comp(pts)
        self.assertTrue(np.alltrue(xx==x))
        self.assertTrue(np.alltrue(yy==y))
        # Confirm the components->pts works
        cpts = to_cpts(x,y)
        self.assertTrue(np.alltrue(cpts==pts))

    def test_conv_1D(self):
        # Start with 1-D arrays of points and the components
        pts = np.array([[i,i*i] for i in range(4)])
        x = np.array([i for i in range(4)])
        y = np.array([i*i for i in range(4)])
        # Confirm the pts->components works
        xx,yy = to_comp(pts)
        self.assertTrue(np.alltrue(xx==x))
        self.assertTrue(np.alltrue(yy==y))
        # Confirm the components->pts works
        cpts = to_cpts(x,y)
        self.assertTrue(np.alltrue(cpts==pts))

    def test_conv_2D(self):
        """Test out conversions for arrays of points"""
        # Start with 2-D arrays of points and the components
        pts = np.array([[[i,j] for j in range(3)] for i in range(4)])
        x = np.array([[i for j in range(3)] for i in range(4)])
        y = np.array([[j for j in range(3)] for i in range(4)])
        # Confirm the pts->components works
        xx,yy = to_comp(pts)
        self.assertTrue(np.alltrue(xx==x))
        self.assertTrue(np.alltrue(yy==y))
        # Confirm the components->pts works
        cpts = to_cpts(x,y)
        self.assertTrue(np.alltrue(cpts==pts))

class TestCpMag(unittest.TestCase):
    """Test for CPoints and numpy arrays of flints"""
    
    def test_cp_mag_float(self):
        """Validate magnitude for a CPoint of floats"""
        a = np.array([3.0, 4.0])
        x = cp_mag(a)
        self.assertIsInstance(x, float)
        self.assertAlmostEqual(x, 5.0)
    
    def test_cp_mag_flint(self):
        """Validate magnitude for a CPoint of flints"""
        a = v_flint([3,4])
        x = cp_mag(a)
        self.assertIsInstance(x, flint)
        self.assertEqual(x, 5.0)

    def test_cp_mag_float_vec(self):
        """Validate magnitude for a 1-d sequence of CPoints of floats"""
        a = np.array([[3.0, 4.0], [-6.0, 8.0]])
        x = cp_mag(a)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(x[0], float)
        self.assertAlmostEqual(x[0], 5.0)
        self.assertAlmostEqual(x[1], 10.0)

    def test_cp_mag_flint_vec(self):
        """Validate magnitude for a 1-d sequence of CPoints of flints"""
        a = v_flint([[3,4],[-6,8]])
        x = cp_mag(a)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(x[0], flint)
        self.assertEqual(x[0], 5.0)
        self.assertEqual(x[1], 10.0)

    def test_cp_mag_float_array(self):
        """Validate magnitude for a 2-d array of CPoints of floats"""
        thi = np.linspace(0,np.pi,3)
        thj = np.linspace(0,np.pi,4)
        thij = np.array([[i+j for j in thj] for i in thi])
        x, y = np.cos(thij), np.sin(thij)
        a = np.array([x.T,y.T]).T
        x = cp_mag(a)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, (3,4))
        self.assertIsInstance(x[0,0], float)
        for i in range(len(x)):
            for j in range(len(x[0])):
                self.assertAlmostEqual(x[i,j], 1.0)

    def test_cp_mag_flint_array(self):
        """Validate magnitude for a 2-d array of CPoints of flints"""
        thi = np.linspace(0,np.pi,3)
        thj = np.linspace(0,np.pi,4)
        thij = np.array([[i+j for j in thj] for i in thi])
        x, y = np.cos(thij), np.sin(thij)
        a = v_flint(np.array([x.T,y.T]).T)
        for i in range(len(a)):
            for j in range(len(a[0])):
                for k in range(len(a[0,0])):
                    a[i,j,k]._grow()
        x = cp_mag(a)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, (3,4))
        self.assertIsInstance(x[0,0], flint)
        for i in range(len(x)):
            for j in range(len(x[0])):
                self.assertEqual(x[i,j], 1.0)
        

class TestCpUnit(unittest.TestCase):
    """Test for CPoints and numpy arrays of flints"""

    def test_cp_unit_float(self):
        """Validate unit vectors for a CPoint of floats"""
        a = np.array([1.0, 1.0])
        u = cp_unit(a)
        self.assertIsInstance(u, np.ndarray)
        self.assertEqual(a.shape, u.shape)
        self.assertAlmostEqual(u[0], 1/np.sqrt(2))
        self.assertAlmostEqual(u[1], 1/np.sqrt(2))

    def test_cp_unit_flint(self):
        """Validate unit vectors for a CPoint of flints"""
        a = v_flint([1.0, 1.0])
        u = cp_unit(a)
        self.assertIsInstance(u, np.ndarray)
        self.assertEqual(a.shape, u.shape)
        self.assertEqual(u[0], 1/np.sqrt(2))
        self.assertEqual(u[1], 1/np.sqrt(2))

    def test_cp_unit_float_array(self):
        """Validate unit vectors for a 2-D array of float CPoints"""
        a = np.array([[[i+1, j+1] for j in range(3)] for i in range(4)], dtype=np.float64)
        u = cp_unit(a)
        self.assertIsInstance(u, np.ndarray)
        self.assertEqual(a.shape, u.shape)
        for i in range(len(u)):
            for j in range(len(u[0])):
                self.assertAlmostEqual(cp_mag(u[i,j]), 1.0)
    
    def test_cp_unit_flint_array(self):
        """Validate unit vectors for a 2-D array of flint CPoints"""
        a = v_flint([[[i+1, j+1] for j in range(3)] for i in range(4)])
        u = cp_unit(a)
        self.assertIsInstance(u, np.ndarray)
        self.assertEqual(a.shape, u.shape)
        for i in range(len(u)):
            for j in range(len(u[0])):
                self.assertEqual(cp_mag(u[i,j]), 1.0)


class TestCpVectorize(unittest.TestCase):
    """Testing the vectorize decorator"""

    @cp_vectorize
    def float_zeros(self, x:float) -> CPoint:
        return np.array([0,0], dtype=np.float64)

    def test_floats_1arg(self):
        """Validate vectorization for single argument functions"""
        # Compare for a float input
        res = self.float_zeros(0)
        target = np.zeros((2,), dtype=np.float64)
        self.assertTrue(np.alltrue(res == target))
        # Compare for a 1-D array input
        res = self.flint_zeros([0,1,2])
        target = np.zeros((3,2), dtype=np.float64)
        self.assertTrue(np.alltrue(res == target))
        # Compare for 2-D array input
        res = self.flint_zeros([[0,1,2,],[3,4,5],[6,7,8],[9,10,11]])
        target = np.zeros((4,3,2), dtype=np.float64)
        self.assertTrue(np.alltrue(res == target))
 
    @cp_vectorize
    def flint_zeros(self, x:float) -> CPoint:
        return np.array([0,0], dtype=np.float64)

    def test_flint_1arg(self):
        """Validate vectorization for single argument functions"""
        # Compare for a float input
        res = self.flint_zeros(0)
        target = np.full((2,), flint(0))
        self.assertTrue(np.alltrue(res == target))
        # Compare for a 1-D array input
        res = self.flint_zeros([0,1,2])
        target = np.full((3,2), flint(0))
        self.assertTrue(np.alltrue(res == target))
        # Compare for 2-D array input
        res = self.flint_zeros([[0,1,2,],[3,4,5],[6,7,8],[9,10,11]])
        target = np.full((4,3,2), flint(0))
        self.assertTrue(np.alltrue(res == target))

    @cp_vectorize
    def zeros_2arg(self, x: float, y: float) -> CPoint:
        return v_flint([0,0])
 
    def test_2arg(self):
        res = self.zeros_2arg(0,0)
        target = np.full((2,), flint(0))
        self.assertTrue(np.alltrue(res == target))
        # Compare for a 1-D array input
        res = self.zeros_2arg([0,1,2],[0,1,2])
        target = np.full((3,2), flint(0))
        self.assertTrue(np.alltrue(res == target))
        # Compare for 2-D array input
        x = [[0,1,2,],[3,4,5],[6,7,8],[9,10,11]]
        res = self.zeros_2arg(x,x)
        target = np.full((4,3,2), flint(0))
        self.assertTrue(np.alltrue(res == target))

    @cp_vectorize(ignore=(2,))
    def zeros_ignore(self, x: float, y: float, a: int) -> CPoint:
        return v_flint([a,a])

    def test_ignore(self):
        res = self.zeros_ignore(0,0,1)
        target = np.full((2,), flint(1))
        self.assertTrue(np.alltrue(res == target))
        # Compare for a 1-D array input
        res = self.zeros_ignore([0,1,2],[0,1,2],2)
        target = np.full((3,2), flint(2))
        self.assertTrue(np.alltrue(res == target))
        # Compare for 2-D array input
        x = [[0,1,2,],[3,4,5],[6,7,8],[9,10,11]]
        res = self.zeros_ignore(x,x,3)
        target = np.full((4,3,2), flint(3))
        self.assertTrue(np.alltrue(res == target))
