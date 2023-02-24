import unittest

import numpy as np

from ..poly import Poly

class TestPoly(unittest.TestCase):

    def test_init(self):
        p = Poly([1.])
        self.assertIsInstance(p, Poly)
        self.assertIsInstance(p.coef, np.ndarray)
        self.assertEqual(len(p.coef), 1)
        p = Poly([1,2,3,4])
        self.assertIsInstance(p, Poly)
        self.assertIsInstance(p.coef, np.ndarray)
        self.assertEqual(len(p.coef), 4)

    def test_repr(self):
        p = Poly([1.])
        self.assertEqual(f'{p}', '1.00')
        p = Poly([0,1])
        self.assertEqual(f'{p}', '1.00*x')
        p = Poly([0,0,0,0,1])
        self.assertEqual(f'{p}', '1.00*x^4')
        p = Poly([1,2,3,4,5])
        self.assertEqual(f'{p}', '1.00 + 2.00*x + 3.00*x^2 + 4.00*x^3 + 5.00*x^4')

    def test_call(self):
        p = Poly([0,0,0.5])
        for i in range(21):
            x = (i/10)
            self.assertEqual(p(x), 0.5*x**2)

    def test_call_nparray(self):
        p = Poly([0,0,0.5])
        x = np.linspace(0,2,21)
        y = p(x)
        yt = 0.5*x**2
        for y1, y2 in zip(y,yt):
            self.assertEqual(y1,y2)

    def test_first_derivative(self):
        p = Poly([0,0,0.5])
        for i in range(21):
            x = (i/10)
            self.assertEqual(p.d(x), x)

    def test_higher_derivatives(self):
        for n in range(2,6):
            p = Poly([0 for _ in range(n)]+[1])
            self.assertEqual(p.d(1,n=n), np.math.factorial(n))

    def test_neg(self):
        c = [1,2,3,4]
        p = -Poly([1,2,3,4])
        for cc, cp in zip(c, p.coef):
            self.assertEqual(-cc, cp)

    def test_add_same_order(self):
        c0 = [1,2,3,4]
        c1 = [4,3,4,3]
        p = Poly(c0) + Poly(c1)
        ct = [5,5,7,7]
        for cc, cp in zip(ct, p.coef):
            self.assertEqual(cc, cp)

    def test_add_different_order(self):
        c0 = [1,2,3,4]
        c1 = [4,3]
        p = Poly(c0) + Poly(c1)
        ct = [5,5,3,4]
        for cc, cp in zip(ct, p.coef):
            self.assertEqual(cc, cp)

    def test_add_cancel_higher_order(self):
        c0 = [1,2,3,4]
        c1 = [4,3,-3,-4]
        p = Poly(c0) + Poly(c1)
        ct = [5,5]
        self.assertEqual(len(p.coef), 2)
        for cc, cp in zip(ct, p.coef):
            self.assertEqual(cc, cp)

    def test_sub(self):
        c0 = [1,2,3,4]
        c1 = [4,3]
        p = Poly(c0) - Poly(c1)
        ct = [-3,-1,3,4]
        for cc, cp in zip(ct, p.coef):
            self.assertEqual(cc, cp)

    def test_mul_scalar(self):
        c = [1,2,3,4]
        p = Poly(c)
        ct = [2,4,6,8]
        for cc, cp in zip(ct, (2*p).coef):
            self.assertEqual(cc, cp)
        for cc, cp in zip(ct, (p*2).coef):
            self.assertEqual(cc, cp)

    def test_mul_poly(self):
        p0, p1 = Poly([1,2,1]), Poly([1,1])
        ct = [1,3,3,1]
        for cc, cp in zip(ct, (p0*p1).coef):
            self.assertEqual(cp, cp)
