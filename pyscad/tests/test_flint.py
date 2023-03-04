import unittest

import numpy as np

from ..flint import flint

class TestInit(unittest.TestCase):
    """Test for the initialization and internal structure of the flint objects"""

    def test_init_int(self):
        """Validate initialization from integers"""
        x = flint(3)
        self.assertIsInstance(x, flint)
        self.assertIsInstance(x.a, np.float64)
        self.assertIsInstance(x.b, np.float64)
        self.assertEqual(x.a, 3)
        self.assertEqual(x.b, 3)

    def test_init_float(self):
        """Validate initialization from floats"""
        x = flint(np.pi)
        self.assertIsInstance(x, flint)
        self.assertIsInstance(x.a, np.float64)
        self.assertIsInstance(x.b, np.float64)
        self.assertEqual(x.a, np.pi)
        self.assertEqual(x.b, np.pi)

    def test_from_interval(self):
        """Validate initialization from an explicitly defined interval"""
        x = flint.from_interval(1, 2)
        self.assertIsInstance(x, flint)
        self.assertIsInstance(x.a, np.float64)
        self.assertIsInstance(x.b, np.float64)
        self.assertEqual(x.a, 1)
        self.assertEqual(x.b, 2)

    def test_init_flint(self):
        """Validate initialization from another flint object"""
        x = flint.from_interval(1, 2)
        y = flint(x)
        self.assertIsInstance(y, flint)
        self.assertIsInstance(y.a, np.float64)
        self.assertIsInstance(y.b, np.float64)
        self.assertEqual(y.a, 1)
        self.assertEqual(y.b, 2)

    def test_grow(self):
        """Validate initialization the internal _grow method"""
        x = flint(1.5)
        x._grow()
        a_target = np.nextafter(np.float64(1.5),-np.inf)
        b_target = np.nextafter(np.float64(1.5), np.inf)
        self.assertEqual(x.a, a_target)
        self.assertEqual(x.b, b_target)

    def test_identical(self):
        """Validate the `identical` comparison method"""
        x = flint(1)
        y = flint(1)
        self.assertTrue(flint.identical(x, y))


class TestComparisonZeroWidth(unittest.TestCase):
    """Test for comparisons with zero width flint objects"""

    def test_gt_float(self):
        """Validate greater than comparison with floats"""
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
        """Validate greater than comparison between flint objects"""
        x = flint(1)
        y = flint(0)
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertTrue(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertFalse(x <= y)

    def test_eq_float(self):
        """Validate equal-to comparison with floats"""
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
        """Validate equal-to comparisons between flint objects"""
        x = flint(1)
        y = flint(1)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y)

    def test_lt_float(self):
        """Validate less than comparison with floats"""
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
        """Validate greater than comparison between flint objects"""
        x = flint(1)
        y = flint(2)
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertFalse(x > y)
        self.assertFalse(x >= y)
        self.assertTrue(x < y)
        self.assertTrue(x <= y)


class TestComparisonNonzeroWidthFloat(unittest.TestCase):
    """Test for comparisons with non-zero width flint objects to floats"""

    def test_below(self):
        """Validate when float is completely below the interval"""
        x = flint.from_interval(1,2)
        y = 0.5
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertTrue(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertFalse(x <= y)

    def test_below_edge(self):
        """Validate when float is on lower edge of the interval"""
        x = flint.from_interval(1,2)
        y = 1
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y)

    def test_include(self):
        """Validate when the float in inside the interval"""
        x = flint.from_interval(1,2)
        y = 1.5
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y)

    def test_above_edge(self):
        """Validate when the float on upper edge of the interval"""
        x = flint.from_interval(1,2)
        y = 2
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y)

    def test_above(self):
        """Validate when float is completely above the interval"""
        x = flint.from_interval(1,2)
        y = 2.5
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertFalse(x > y)
        self.assertFalse(x >= y)
        self.assertTrue(x < y)
        self.assertTrue(x <= y)


class TestComparisonNonzeroWidth(unittest.TestCase):
    """Test between two non-zero width flint objects"""

    def test_below(self):
        """Validate comparison operators for lhs interval completely below the other"""
        x = flint.from_interval(1, 2)
        y = flint.from_interval(0, 0.5)
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertTrue(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertFalse(x <= y)

    def test_below_touch(self):
        """Validate comparison operators for lhs interval just touches the other from below"""
        x = flint.from_interval(1, 2)
        y = flint.from_interval(0, 1)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y) 

    def test_below_overlap(self):
        """Validate comparison operators for lhs interval below but overlapping the other"""
        x = flint.from_interval(1, 2)
        y = flint.from_interval(0.5, 1.5)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y) 

    def test_internal(self):
        """Validate comparison operators when lhs interval is completely inside the other"""
        x = flint.from_interval(1, 2)
        y = flint.from_interval(1.3, 1.7)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y) 

    def test_external(self):
        """Validate comparison operators when lhs interval completely contains the other"""
        x = flint.from_interval(1, 2)
        y = flint.from_interval(0.5, 2.5)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y) 

    def test_above_overlap(self):
        """Validate comparison operators for lhs interval above but overlapping the other"""
        x = flint.from_interval(1, 2)
        y = flint.from_interval(1.5, 2.5)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y) 

    def test_above_touch(self):
        """Validate comparison operators for lhs interval just touches the other from above"""
        x = flint.from_interval(1, 2)
        y = flint.from_interval(2, 3)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x > y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertTrue(x <= y) 

    def test_above(self):
        """Validate comparison operators for lhs interval completely below the other"""
        x = flint.from_interval(1, 2)
        y = flint.from_interval(2.5, 3.5)
        self.assertFalse(x == y)
        self.assertTrue(x != y)
        self.assertFalse(x > y)
        self.assertFalse(x >= y)
        self.assertTrue(x < y)
        self.assertTrue(x <= y) 


class TestArithmetic(unittest.TestCase):
    """Test arithmetic operators for flint objects"""

    def test_add(self):
        """Validate addition"""
        x = flint(1)
        t = flint(2)
        t._grow()
        self.assertTrue(flint.identical(x+1, t))
        self.assertTrue(flint.identical(1+x, t))
        x += 1
        self.assertTrue(flint.identical(x, t))

    def test_sub(self):
        """Validate subtraction"""
        x = flint(2)
        y = flint(1)
        t = flint(1)
        t._grow()
        self.assertTrue(flint.identical(x-1, t))
        self.assertTrue(flint.identical(2-y, t))
        x -= 1
        self.assertTrue(flint.identical(x, t))

    def test_mul(self):
        """Validate multiplication"""
        x = flint(2)
        t = flint(6)
        t._grow()
        self.assertTrue(flint.identical(x*3, t))
        self.assertTrue(flint.identical(3*x, t))
        x *= 3
        self.assertTrue(flint.identical(x, t))

    def test_div(self):
        """Validate division"""
        x = flint(6)
        y = flint(3)
        t = flint(2)
        t._grow()
        self.assertTrue(flint.identical(x/3, t))
        self.assertTrue(flint.identical(6/y, t))
        x /= 3
        self.assertTrue(flint.identical(x, t))

    def test_frac(self):
        """Validate creation from a fraction"""
        x = flint.frac(1,3)
        y = flint(1)/3
        z = 1/flint(3)
        self.assertTrue(flint.identical(x, y))
        self.assertTrue(flint.identical(x, z))


class TestUnary(unittest.TestCase):
    """Test the unary operators"""

    def test_neg(self):
        """Validate negation"""
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
        """Validate explicit positive sign"""
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
        """Validate absolute value"""
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

    def test_sqrt(self):
        """Validate square root"""
        x = flint(1)
        x.a = 1.0
        x.v = 4.0
        x.b = 9.0
        y = x.sqrt()
        self.assertEqual(y.a, np.nextafter(1.0, -np.inf))
        self.assertEqual(y.v, 2.0)
        self.assertEqual(y.b, np.nextafter(3.0, np.inf))
        x = flint(1)
        x.a = -1.0
        x.v = 1.0
        x.b = 4.0
        y = x.sqrt()
        self.assertEqual(y.a, 0.0)
        self.assertEqual(y.v, 1.0)
        self.assertEqual(y.b, np.nextafter(2.0, np.inf))
        x = flint(1)
        x.a = -1.0
        x.v = -1.0
        x.b = 1.0
        y = x.sqrt()
        self.assertEqual(y.a, 0.0)
        self.assertEqual(y.v, 0.0)
        self.assertEqual(y.b, np.nextafter(1.0, np.inf))
        x = flint(-1)
        with self.assertRaises(ValueError):
            x.sqrt()
    
