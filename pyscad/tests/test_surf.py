import unittest

import numpy as np

from ..surf import *

class MissingMethods(ParaSurf):
    """Only hear to validate errors"""
    pass


class TestSurfProp(unittest.TestCase):
    """Test suite for general properties of parametric surfaces"""

    def test_error(self):
        """Validate that exception is thrown if methods aren't yet defined"""
        mm = MissingMethods()
        with self.assertRaises(NotImplementedError):
            mm(1.0, 1.0)
