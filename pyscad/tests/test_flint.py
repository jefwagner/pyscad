import unittest

import numpy as np

from ..flint import flint

class TestInit(unittest.TestCase):

    def test_init_int(self):
        x = flint(3)
        self.assertIsInstance(x, flint)
        self.assertIsInstance(x.a, np.float64)
        self.assertIsInstance(x.b, np.float64)
        self.assertEqual(x.a, 3)
        self.assertEqual(x.b, 3)

    def test_init_float(self):
        x = flint(np.pi)
        self.assertIsInstance(x, flint)
        self.assertIsInstance(x.a, np.float64)
        self.assertIsInstance(x.b, np.float64)
        self.assertEqual(x.a, np.pi)
        self.assertEqual(x.b, np.pi)

    def test_from_interval(self):
        x = flint.from_interval(1, 2)
        self.assertIsInstance(x, flint)
        self.assertIsInstance(x.a, np.float64)
        self.assertIsInstance(x.b, np.float64)
        self.assertEqual(x.a, 1)
        self.assertEqual(x.b, 2)

    def test_init_flint(self):
        x = flint.from_interval(1, 2)
        y = flint(x)
        self.assertIsInstance(y, flint)
        self.assertIsInstance(y.a, np.float64)
        self.assertIsInstance(y.b, np.float64)
        self.assertEqual(y.a, 1)
        self.assertEqual(y.b, 2)

    def test_grow(self):
        x = flint(1.5)
        x._grow()
        a_target = np.nextafter(np.float64(1.5),-np.inf)
        b_target = np.nextafter(np.float64(1.5), np.inf)
        self.assertEqual(x.a, a_target)
        self.assertEqual(x.b, b_target)

    def test_identical(self):
        x = flint(1)
        y = flint(1)
        self.assertTrue(flint.identical(x, y))


class TestComparisonZeroWidth(unittest.TestCase):

    def test_gt_float(self):
        x = flint(1)
        y = 0
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertTrue(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertFalse(x <= y)
        x = 1
        y = flint(0)
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertTrue(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertFalse(x <= y)

    def test_gt_flint(self):
        x = flint(1)
        y = flint(0)
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertTrue(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertFalse(x <= y)

    def test_eq_float(self):
        x = flint(1)
        y = 1
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y)
        x = 1
        y = flint(1)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y)

    def test_eq_flint(self):
        x = flint(1)
        y = flint(1)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y)

    def test_lt_float(self):
        x = flint(1)
        y = 2
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertFalse(x > y)
        self.assertFalse(x >= y)
        self.assertTrue(x < y)
        self.assertTrue(x <= y)
        x = 1
        y = flint(2)
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertFalse(x > y)
        self.assertFalse(x >= y)
        self.assertTrue(x < y)
        self.assertTrue(x <= y)

    def test_lt_flint(self):
        x = flint(1)
        y = flint(2)
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertFalse(x > y)
        self.assertFalse(x >= y)
        self.assertTrue(x < y)
        self.assertTrue(x <= y)


class TestComparisonNonzeroWidthFloat(unittest.TestCase):

    def test_below(self):
        x = flint.from_interval(1,2)
        y = 0.5
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertTrue(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertFalse(x <= y)

    def test_below_edge(self):
        x = flint.from_interval(1,2)
        y = 1
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y)

    def test_include(self):
        x = flint.from_interval(1,2)
        y = 1.5
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y)

    def test_above_edge(self):
        x = flint.from_interval(1,2)
        y = 2
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y)

    def test_above(self):
        x = flint.from_interval(1,2)
        y = 2.5
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertFalse(x > y)
        self.assertFalse(x >= y)
        self.assertTrue(x < y)
        self.assertTrue(x <= y)


class TestComparisonNonzeroWidth(unittest.TestCase):

    def test_below(self):
        x = flint.from_interval(1, 2)
        y = flint.from_interval(0, 0.5)
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertTrue(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertFalse(x <= y)

    def test_below_touch(self):
        x = flint.from_interval(1, 2)
        y = flint.from_interval(0, 1)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y) 

    def test_below_overlap(self):
        x = flint.from_interval(1, 2)
        y = flint.from_interval(0.5, 1.5)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y) 

    def test_internal(self):
        x = flint.from_interval(1, 2)
        y = flint.from_interval(1.3, 1.7)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y) 

    def test_external(self):
        x = flint.from_interval(1, 2)
        y = flint.from_interval(0.5, 2.5)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y) 

    def test_above_overlap(self):
        x = flint.from_interval(1, 2)
        y = flint.from_interval(1.5, 2.5)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y) 

    def test_above_touch(self):
        x = flint.from_interval(1, 2)
        y = flint.from_interval(2, 3)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y) 

    def test_above(self):
        x = flint.from_interval(1, 2)
        y = flint.from_interval(2.5, 3.5)
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertFalse(x > y)
        self.assertFalse(x >= y)
        self.assertTrue(x < y)
        self.assertTrue(x <= y) 


class TestArithmetic(unittest.TestCase):

    def test_add(self):
        x = flint(1)
        t = flint(2)
        t._grow()
        self.assertTrue(flint.identical(x+1, t))
        self.assertTrue(flint.identical(1+x, t))
        x += 1
        self.assertTrue(flint.identical(x, t))

    def test_sub(self):
        x = flint(2)
        y = flint(1)
        t = flint(1)
        t._grow()
        self.assertTrue(flint.identical(x-1, t))
        self.assertTrue(flint.identical(2-y, t))
        x -= 1
        self.assertTrue(flint.identical(x, t))

    def test_mul(self):
        x = flint(2)
        t = flint(6)
        t._grow()
        self.assertTrue(flint.identical(x*3, t))
        self.assertTrue(flint.identical(3*x, t))
        x *= 3
        self.assertTrue(flint.identical(x, t))

    def test_div(self):
        x = flint(6)
        y = flint(3)
        t = flint(2)
        t._grow()
        self.assertTrue(flint.identical(x/3, t))
        self.assertTrue(flint.identical(6/y, t))
        x /= 3
        self.assertTrue(flint.identical(x, t))

    def test_frac(self):
        x = flint.frac(1,3)
        y = flint(1)/3
        z = 1/flint(3)
        self.assertTrue(flint.identical(x, y))
        self.assertTrue(flint.identical(x, z))


class TestUnary(unittest.TestCase):

    def test_neg(self):
        x = flint(1)
        y = flint(-1)
        self.assertTrue(flint.identical(-x, y))
        self.assertTrue(flint.identical(x, -y))
        x = flint.from_interval(-0.5, 0.5)
        self.assertTrue(flint.identical(x, -x))
        x = flint.from_interval(-1, 2)
        y = flint.from_interval(-2, 1)
        self.assertTrue(flint.identical(-x, y))
        self.assertTrue(flint.identical(x, -y))
        x = flint.from_interval(1, 2)
        y = flint.from_interval(-2, -1)
        self.assertTrue(flint.identical(-x, y))
        self.assertTrue(flint.identical(x, -y))

    def test_pos(self):
        x = flint(1)
        self.assertTrue(flint.identical(x, +x))
        x = flint(-1)
        self.assertTrue(flint.identical(x, +x))
        x = flint.from_interval(-1, 1)
        self.assertTrue(flint.identical(x, +x))
        x = flint.from_interval(1, 2)
        self.assertTrue(flint.identical(x, +x))
        x = flint.from_interval(-2, -1)
        self.assertTrue(flint.identical(x, +x))

    def test_abs(self):
        x = flint(-1)
        y = flint(1)
        t = flint(1)
        self.assertTrue(flint.identical(abs(x), t))
        self.assertTrue(flint.identical(abs(y), t))
        x = flint.from_interval(-1, 2)
        x = abs(x)
        self.assertEqual(x.a, 0)
        self.assertEqual(x.b, 2)
        self.assertEqual(x.v, 0.5)
        x = flint.from_interval(-2, 1)
        x = abs(x)
        self.assertEqual(x.a, 0)
        self.assertEqual(x.b, 2)
        self.assertEqual(x.v, 0.5)
        x = flint.from_interval(1, 2)
        t = flint(x)
        x = abs(x)
        self.assertTrue(flint.identical(x, t))
        x = flint.from_interval(-2, -1)
        t = flint.from_interval(1, 2)
        x = abs(x)
        self.assertTrue(flint.identical(x, t))

