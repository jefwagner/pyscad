import unittest

import numpy as np

from ..flint import v_flint
from ..curves import SpaceCurve
from ..bspline import BSpline
from .test_curves import simple_basis, simple_basis_d1, simple_basis_d2

class TestBSpline(unittest.TestCase):
    """Test the basis-spline object"""

    def setUp(self):
        """We will be testing against simple basis on 0123 knot vector"""
        p = 2
        c = [[1,-1]]
        t = [0,1,2,3]
        self.bs = BSpline(c,p,t)

    def test_init(self):
        """Validate the initialization works correctly"""
        self.assertIsInstance(self.bs, SpaceCurve)
        self.assertIsInstance(self.bs, BSpline)
        self.assertEqual(self.bs.p, 2)
        self.assertIsInstance(self.bs.c, np.ndarray)
        with self.assertRaises(ValueError):
            bad = BSpline(self.bs.c, self.bs.p, [0,1])
    
    def test_call_float(self):
        for tval in np.linspace(0,3,10):
            a = self.bs(tval)
            b = simple_basis(tval)
            # value were close, but not close enough, needed to grow
            a[0]._grow()
            a[1]._grow()
            self.assertEqual(a[0], b)
            self.assertEqual(a[1], -b)
