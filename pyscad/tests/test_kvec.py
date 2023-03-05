import unittest
import numpy as np

from ..flint import v_flint
from ..kvec import KnotVector

# A single spline basis function is calculated for degree 2 with the knot-vector
# (0,1,2,3).
# $$
# B_{2,0}(t) = \begin{cases}
#     t^2/2 &\text{for}\quad 0 \le t < 1, \\
#     (-2t^2+6t-3)/2 \quad&\text{for} 1 \le t < 2, \\
#     (3-t)^2/2 \quad&\text{for} 2 \le t < 3.
# $$
def simple_basis(t:float) -> float:
    """Simple basis function of degree 2 for knot-vector (0,1,2,3)"""
    if t < 1:
        return 0.5*t*t
    elif t < 2:
        return 0.5*(-2*t*t+6*t-3)
    else:
        return 0.5*(3-t)*(3-t)

def simple_basis_d1(t:float) -> float:
    """first derivative of the basis function of degree 2 for knot-vector (0,1,2,3)"""
    if t < 1:
        return 1.0*t
    elif t < 2:
        return -2.0*t+3
    else:
        return t-3.0

def simple_basis_d2(t:float) -> float:
    """second derivative of the basis function of degree 2 for knot-vector (0,1,2,3)"""
    if t < 1:
        return 1.0
    elif t < 2:
        return -2.0
    else:
        return 1.0

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

    def test_deboor(self):
        """Validate the de Boor algorithm for a simple case"""
        p = 2
        c = [1.0]
        tv = KnotVector([0,1,2,3])
        for t in np.linspace(0,3,30):
            basis_spline = tv.deboor(c,p,t)
            basis_func = simple_basis(t)
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