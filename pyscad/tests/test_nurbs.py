import unittest

import numpy as np

from ..flint import v_flint
from ..cpoint import cp_mag
from ..curves import SpaceCurve
from ..nurbs import NurbsCurve

# For all test we will be doing a quarter circle nurbs curve
qc_c = [[1,0],[1,1],[0,1]]
qc_w = [1,1/np.sqrt(2),1]
qc_p = 2
qc_t = [0,0,0,1,1,1]
qc = NurbsCurve(qc_c, qc_w, qc_p, qc_t)

class TestInit(unittest.TestCase):
    """Test that the class and members are created correctly"""

    def test_bad_init(self):
        """Validate exceptions if an invalid nurbs is given"""
        # Length of control points and weights don't match
        with self.assertRaises(ValueError):
            ns = NurbsCurve(qc_c, [1], qc_p, qc_t)
        # Incorrect length of the knot-vector
        with self.assertRaises(ValueError):
            ns = NurbsCurve(qc_c, qc_w, qc_p, [0,1])

    def test_init(self):
        """Validate the class types and internal members"""
        ns = NurbsCurve(qc_c,qc_w,qc_p,qc_t)
        self.assertIsInstance(ns, SpaceCurve)
        self.assertIsInstance(ns, NurbsCurve)
        self.assertEqual(len(ns.c), len(ns.w))
        self.assertEqual(ns.p, 2)
        self.assertIsInstance(ns.c, np.ndarray)
        self.assertEqual(ns.c.shape, (3,2))
        self.assertIsInstance(ns.w, np.ndarray)
        self.assertEqual(ns.w.shape, (3,1))


class TestEval(unittest.TestCase):
    """Test evaluation of the curve and derivatives"""

    def setUp(self):
        """All curves will use a quarter circle"""
        self.qc = NurbsCurve(qc_c, qc_w, qc_p, qc_t)

    def test_call_scalar(self):
        """Validate evaluating the curve at scalar parameter values"""
        for t in np.linspace(0,1,10):
            # Confirm all points have a length of 1
            p = self.qc(t)
            self.assertEqual(1.0, cp_mag(p))
    
    def test_call_vec(self):
        """Validate evaluating the curve at a 1-D vector of parameter values"""
        t = np.linspace(0,1,10)
        # Confirm all points have a length of 1
        pts = self.qc(t)
        for p in pts:
            self.assertEqual(1.0, cp_mag(p))

    def test_tangent_vec(self):
        """Validate the first derivative by verifying the tangent"""
        t = np.linspace(0,1,10)
        # The tangent should always be 90 degrees off the radius for a circle
        pts = self.qc(t)
        tans = self.qc.tangent(t)
        for pt, tan in zip(pts, tans):
            self.assertEqual(pt[0],tan[1])
            self.assertEqual(pt[1],-tan[0])

    def test_curvature_vec(self):
        """Validate the second derivative by verifying curvature"""
        t = np.linspace(0,1,10)
        # Curvature of a unit radius curve is always 1
        kappas = self.qc.curvature(t)
        for kappa in kappas:
            self.assertEqual(1.0, kappa)
