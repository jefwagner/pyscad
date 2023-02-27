import unittest

import numpy as np

from ..nurbs import *

class TestKnotVector(unittest.TestCase):

    def test_init(self):
        tv = KnotVector([0,0,0,0.5,1,1,1])
        self.assertIsInstance(tv, KnotVector)
        self.assertIsInstance(tv.t, np.ndarray)
        self.assertEqual(len(tv.t), 7)

    def test_init_error(self):
        with self.assertRaises(ValueError):
            tv = KnotVector([1,0])
    
    def test_kminmax(self):
        tv = KnotVector([0.1*i for i in range(11)])
        self.assertEqual(tv.kmin, 0)
        self.assertEqual(tv.kmax, 9)
        tv = KnotVector([0,0,0,0.5,1,1,1])
        self.assertEqual(tv.kmin, 2)
        self.assertEqual(tv.kmax, 3)

    def test_len(self):
        for m in range(3,10):
            tv = KnotVector([(1/(m-1))*i for i in range(m)])
            self.assertEqual(len(tv), m)

    def test_getitem(self):
        tv = KnotVector([0.1*i for i in range(11)])
        for i in range(11):
            self.assertEqual(tv[i], 0.1*i)

    def test_iter(self):
        tv = KnotVector([0.1*i for i in range(11)])
        for i, t in enumerate(tv):
            self.assertEqual(t, 0.1*i)
    
    def test_k(self):
        tv = KnotVector([0.1*i for i in range(11)])
        for i in range(10):
            self.assertEqual(tv.k(0.1*i+0.05), i)

    def test_k_oob(self):
        tv = KnotVector([0.1*i for i in range(11)])
        self.assertEqual(tv.k(-0.01), 0)
        self.assertEqual(tv.k(1.0), 9)
        tv = KnotVector([0,0,0,0.5,1,1,1])
        self.assertEqual(tv.k(-0.01), 2)
        self.assertEqual(tv.k(1.0), 3)

    def test_k_nparray(self):
        tv = KnotVector([0.1*i for i in range(11)])
        k = tv.k([0.1*i+0.05 for i in range(10)])
        for i, kk in enumerate(k):
            self.assertEqual(i, kk)

    # def test_deboor(self):
    #     p = 3
    #     c = [0,1,0]
    #     tv = KnotVector([0,0,0,0.5,1,1,1])
    #     for x in np.linspace(0,1,11):
    #         print(x, tv.deboor(p,c,x))

    # def test_deboor_nparray(self):
    #     p = 3
    #     c = [0,1,0]
    #     tv = KnotVector([0,0,0,0.5,1,1,1])
    #     x = np.linspace(0,1,11)
    #     y = tv.deboor(p,c,x)
    #     print(y)


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

