import unittest

import numpy as np

from ..flint import v_flint
from ..curves import *

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
        self.assertEqual(c0, 2)
        print(cp)
        self.assertEqual(cn, cp)


