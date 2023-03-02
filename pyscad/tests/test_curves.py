import unittest

import numpy as np

from ..curves import *

class TestKnotVector(unittest.TestCase):
    """A test suite for the a knot-vector object"""

    def test_init(self):
        """Make sure the object is initialized properly"""
        tv = KnotVector([0,0,0,0.5,1,1,1])
        self.assertIsInstance(tv, KnotVector)
        self.assertIsInstance(tv.t, np.ndarray)
        self.assertEqual(len(tv.t), 7)

    def test_init_error(self):
        """Make sure we get an error if we violate the non-decreasing condition"""
        with self.assertRaises(ValueError):
            tv = KnotVector([1,0])
    
    def test_kminmax(self):
        """Validate that we properly identify the first and last non-zero interval"""
        tv = KnotVector([0.1*i for i in range(11)])
        self.assertEqual(tv.kmin, 0)
        self.assertEqual(tv.kmax, 9)
        tv = KnotVector([0,0,0,0.5,1,1,1])
        self.assertEqual(tv.kmin, 2)
        self.assertEqual(tv.kmax, 3)

    def test_len(self):
        """Validate the len method"""
        for m in range(3,10):
            tv = KnotVector([(1/(m-1))*i for i in range(m)])
            self.assertEqual(len(tv), m)

    def test_getitem(self):
        """Validate the indexing for the knot-vector works"""
        tv = KnotVector([0.1*i for i in range(11)])
        for i in range(11):
            self.assertEqual(tv[i], 0.1*i)

    def test_iter(self):
        """Validate we can iterate over the knot-vector"""
        tv = KnotVector([0.1*i for i in range(11)])
        for i, t in enumerate(tv):
            self.assertEqual(t, 0.1*i)
    
    def test_k(self):
        """Validate we can identify the interval that contains parametric value"""
        tv = KnotVector([0.1*i for i in range(11)])
        for i in range(10):
            self.assertEqual(tv.k(0.1*i+0.05), i)

    def test_k_oob(self):
        """Validate how we handle out of bounds indexing"""
        tv = KnotVector([0.1*i for i in range(11)])
        self.assertEqual(tv.k(-0.01), 0)
        self.assertEqual(tv.k(1.0), 9)
        tv = KnotVector([0,0,0,0.5,1,1,1])
        self.assertEqual(tv.k(-0.01), 2)
        self.assertEqual(tv.k(1.0), 3)

    def test_k_nparray(self):
        """Validate that the indexing works for ArrayLike inputs"""
        tv = KnotVector([0.1*i for i in range(11)])
        k = tv.k([0.1*i+0.05 for i in range(10)])
        for i, kk in enumerate(k):
            self.assertEqual(i, kk)

    # The deboor's algorithm uses a knot-vector to evaluate splines. To test this we
    # will use the hand derived basis function or degree 2 for the knot-vector (0,1,2,3)
    # $$
    # B_{2,0}(t) = \begin{cases}
    #     t^2/2 &\text{for}\quad 0 \le t < 1, \\
    #     (-2t^2+6t-3)/2 \quad&\text{for} 1 \le t < 2, \\
    #     (3-t)^2/2 \quad&\text{for} 2 \le t < 3.
    # $$
    def simple_basis(self, t:float) -> float:
        """Simple basis function of degree 2 for knot-vector (0,1,2,3)"""
        if t < 1:
            return 0.5*t*t
        elif t < 2:
            return 0.5*(-2*t*t+6*t-3)
        else:
            return 0.5*(3-t)*(3-t)

    def test_deboor(self):
        """Validate the de Boor algorithm for a simple case"""
        p = 2
        c = [1.0]
        tv = KnotVector([0,1,2,3])
        for t in np.linspace(0,3,30):
            basis_spline = tv.deboor(c,p,t)
            basis_func = self.simple_basis(t)
            self.assertAlmostEqual(basis_spline, basis_func)

    def test_d_points(self):
        """Validate the derivative control points for a simple case"""
        p = 2
        c = [1.0]
        tv = KnotVector([0,1,2,3])
        d = tv.d_points(c, p, 1)
        self.assertAlmostEqual(d[0], 1.0)
        self.assertAlmostEqual(d[1], -1.0)


class TestBSpline(unittest.TestCase):

    def test_init(self):
        p = 2
        c = [0,1]
        t = [0,0,0,0,1]
        with self.assertRaises(ValueError):
            bs = BSpline(c,p,[0,1])
        bs = BSpline(c,p,t)
        self.assertIsInstance(bs, SpaceCurve)
        self.assertIsInstance(bs, BSpline)
        self.assertEqual(bs.p, 2)
        self.assertIsInstance(bs.c, np.ndarray)


class TestNurbsCurve(unittest.TestCase):

    def test_init(self):
        p = 2
        c = [[1,0],[1,1],[0,1]]
        w = [1,1/np.sqrt(2),1]
        t = [0,0,0,1,1,1]
        with self.assertRaises(ValueError):
            ns = NurbsCurve(c,[1],p,t)
        with self.assertRaises(ValueError):
            ns = NurbsCurve(c,w,p,[0,1])
        ns = NurbsCurve(c,w,p,t)
        self.assertIsInstance(ns, SpaceCurve)
        self.assertIsInstance(ns, NurbsCurve)
        self.assertEqual(len(ns.c), len(ns.w))
        self.assertEqual(ns.p, 2)
        self.assertIsInstance(ns.c, np.ndarray)
        self.assertIsInstance(ns.w, np.ndarray)

