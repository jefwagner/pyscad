import unittest

import numpy as np

from ..flint import v_flint
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

    def test_d_cpts(self):
        """Validate the derivative control points for a simple case"""
        p = 2
        c = [1.0]
        tv = KnotVector([0,1,2,3])
        d = tv.d_cpts(c, p)
        self.assertEqual(len(d), 2)
        self.assertAlmostEqual(d[0], 1.0)
        self.assertAlmostEqual(d[1], -1.0)

    def test_d_cpts_list(self):
        """Validate the list of derivative control points for a simple case"""
        p = 2
        c = [1.0]
        tv = KnotVector([0,1,2,3])
        pts_list = tv.d_cpts_list(c, p, 2)
        self.assertEqual(len(pts_list), 3)
        for i, c in enumerate(pts_list):
            self.assertEqual(len(c), i+1)
        d0, d1, d2 = pts_list
        self.assertAlmostEqual(d0[0], c[0])
        self.assertAlmostEqual(d1[0], 1.0)
        self.assertAlmostEqual(d1[1], -1.0)
        self.assertAlmostEqual(d2[0], 1.0)
        self.assertAlmostEqual(d2[1], -2.0)
        self.assertAlmostEqual(d2[2], 1.0)

class MissingMethods(SpaceCurve):
    """Only hear to validate errors"""
    pass

class Parabola(SpaceCurve):
    """Give a parabola in 2-D"""

    def __call__(self, x):
        """Evalute the parabola"""
        x = v_flint(x)
        y = x*x
        return np.array([x,y]).T        

    def d(self, x, n):
        """Evaluate the nth order derivatives of the parabola"""
        if n == 1:
            y = 2*v_flint(x)
            x = np.ones_like(y)
            return np.array([x,y]).T
        elif n == 2:
            _x = v_flint(x)
            x = np.zeros_like(_x)
            y = 2*np.ones_like(_x)
            return np.array([x,y]).T
        else:
            _x = v_flint(x)
            x = np.zeros_like(_x)
            y = np.zeros_like(_x)
            return np.array([x,y]).T


class TestSpaceCurve(unittest.TestCase):
    """Test the generic space-curve functions"""

    def test_error(self):
        """Validate that if methods aren't """
        mm = MissingMethods()
        with self.assertRaises(NotImplementedError):
            mm(1.0)
        with self.assertRaises(NotImplementedError):
            mm.d(1.0, 1)
        with self.assertRaises(NotImplementedError):
            mm.d_list(1.0, 2)
        with self.assertRaises(NotImplementedError):
            mm.arclen(0,1)

    def test_d_list(self):
        p = Parabola()
        d, t, n = p.d_list(0.5, 2)
        self.assertEqual(len(d), 2)
        self.assertEqual(d[0], 0.5)
        self.assertEqual(d[1], 0.25)
        self.assertEqual(len(t), 2)
        self.assertEqual(t[0], 1)
        self.assertEqual(t[1], 1)
        self.assertEqual(len(n), 2)
        self.assertEqual(n[0], 0)
        self.assertEqual(n[1], 2)

    def test_arclen(self):
        p = Parabola()
        al = p.arclen(0,1)
        target = 0.5*np.sqrt(5) + 0.25*np.log(2+np.sqrt(5))
        self.assertAlmostEqual(al, target)

    def test_tanget(self):
        p = Parabola()
        t = p.tangent(0.5)
        self.assertEqual(len(t), 2)
        self.assertEqual(t[0], 1/np.sqrt(2))
        self.assertEqual(t[1], 1/np.sqrt(2))

    def test_tangent_ufunc(self):
        p = Parabola()
        t0, t5, t1 = p.tangent([0,0.5,1])
        self.assertEqual(len(t0), 2)
        self.assertEqual(t0[0], 1)
        self.assertEqual(t0[1], 0)
        self.assertEqual(len(t5), 2)
        self.assertEqual(t5[0], 1/np.sqrt(2))
        self.assertEqual(t5[1], 1/np.sqrt(2))
        self.assertEqual(len(t1), 2)
        self.assertEqual(t1[0], 1/np.sqrt(5))
        self.assertEqual(t1[1], 2/np.sqrt(5))

    def test_curvature(self):
        p = Parabola()
        self.assertEqual(p.curvature(0), 2 )

    def test_curvature_ufunc(self):
        p = Parabola()
        cn, c0, cp = p.curvature([-1,0,1])
        self.assertEqual(c0, 2 )
        print(cp)
        self.assertEqual(cn, cp)


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

