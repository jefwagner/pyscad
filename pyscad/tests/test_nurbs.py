import unittest

import numpy as np

from ..flint import v_flint
from ..curves import SpaceCurve
from ..nurbs import NurbsCurve

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
