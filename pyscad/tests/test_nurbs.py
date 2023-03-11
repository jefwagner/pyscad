import unittest

import numpy as np

from ..flint import flint, v_flint
from ..cpoint import cp_mag
from ..curves import ParaCurve
from ..surf import ParaSurf
from ..nurbs import NurbsCurve, NurbsSurf

# For all curve test we will be doing a NURBS quarter circle
qc_c = [[1,0],[1,1],[0,1]]
qc_w = [1,1/np.sqrt(2),1]
qc_p = 2
qc_t = [0,0,0,1,1,1]
qc = NurbsCurve(qc_c, qc_w, qc_p, qc_t)

# For all surface test we will be doing a section of torus
R = 2
a = 1
ts_c = [[[R+a,0,0],[R+a,0,a],[R,0,a]],
        [[R+a,R+a,0], [R+a,R+a,a],[R,R,a]],
        [[0,R+a,0],[0,R+a,a],[0,R,a]]]
ts_w = [[1,1/np.sqrt(2),1],[1/np.sqrt(2),1/2,1/np.sqrt(2)],[1,1/np.sqrt(2),1]]
ts_pu, ts_pv = 2, 2
ts_tu, ts_tv = [0,0,0,1,1,1], [0,0,0,1,1,1]
ts = NurbsSurf(ts_c, ts_w, ts_pu, ts_pv, ts_tu, ts_tv)

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

    def test_curve(self):
        """Validate the class types and internal members for a curve"""
        ns = NurbsCurve(qc_c,qc_w,qc_p,qc_t)
        self.assertIsInstance(ns, ParaCurve)
        self.assertIsInstance(ns, NurbsCurve)
        self.assertEqual(len(ns.c), len(ns.w))
        self.assertEqual(ns.p, 2)
        self.assertIsInstance(ns.c, np.ndarray)
        self.assertEqual(ns.c.shape, (3,2))
        self.assertIsInstance(ns.w, np.ndarray)
        self.assertEqual(ns.w.shape, (3,1))

    def test_surf(self):
        """Validate the class types and internal members for a surface"""
        raise NotImplementedError()

class TestCurveEval(unittest.TestCase):
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


class TestSurfEval(unittest.TestCase):
    """Test the evaluation of the surface and derivatives"""

    def setUp(self):
        """All test will be done with a torus section"""
        self.tc = NurbsSurf(ts_c, ts_w, ts_pu, ts_pv, ts_tu, ts_tv)

    def test_call_scalar(self):
        """Validate evaluation of the surface for scalar arguments"""
        uvpts = [[0.0,0.0],
                 [0.5,0.0],
                 [0.0,0.5]]
        tpts = [[3.0, 0, 0],
                [3.0/np.sqrt(2),3.0/np.sqrt(2),0],
                [2+1/np.sqrt(2),0,1/np.sqrt(2)]]
        for uv, tpt in zip(uvpts, tpts):
            cpt = self.tc(*uv)
            self.assertTrue(np.allclose(tpt, cpt.astype(float)))
        
    def test_call_array(self):
        """Validate evaluation of the surface for array arguments"""
        u = [0,0.5]
        v = [0,0.5]
        U,V = np.meshgrid(u,v)
        tpts = [[[3.0,0,0], # (0,0)
                [3.0/np.sqrt(2),3.0/np.sqrt(2),0]], #(0.5,0)
                [[2+1/np.sqrt(2),0,1/np.sqrt(2)], #(0,0.5)
                 [np.sqrt(2)+1/2,np.sqrt(2)+1/2,1/np.sqrt(2)]]] # (0.5,0.5)
        cpts = self.tc(U,V).astype(float)
        self.assertTrue(np.allclose(tpts, cpts))
    
    def test_normal_scalar(self):
        """Validate evaluation of surface normal for scalar arguments"""
        uvpts = [[0.0,0.0],
                 [0.5,0.0],
                 [0.0,0.5]]
        tns = [[1.0, 0, 0],
                [1/np.sqrt(2),1/np.sqrt(2),0],
                [1/np.sqrt(2),0,1/np.sqrt(2)]]
        for uv, tn in zip(uvpts, tns):
            cn = self.tc.normal(*uv)
            self.assertTrue(np.allclose(tn, cn.astype(float)))

    def test_normal_array(self):
        """Validate evaluation of surface normal for scalar arguments"""
        u = [0,0.5]
        v = [0,0.5]
        U,V = np.meshgrid(u,v)
        tns = [[[1.0,0,0], # (0,0)
                [1/np.sqrt(2),1/np.sqrt(2),0]], #(0.5,0)
                [[1/np.sqrt(2),0,1/np.sqrt(2)], #(0,0.5)
                 [1/2,1/2,1/np.sqrt(2)]]] # (0.5,0.5)
        cns = self.tc.normal(U,V).astype(float)
        self.assertTrue(np.allclose(tns, cns))

    def test_mean_curvature(self):
        """Validate evaluation of mean curvature"""
        uvpts = [[0,0],[1,0],[0,1],[1,1]]
        Hs = [
            flint.frac(-2,3),
            flint.frac(-2,3),
            flint.frac(-1,2),
            flint.frac(-1,2)
        ]
        for uv, H in zip(uvpts,Hs):
            k_mean = self.tc.k_mean(*uv)
            self.assertEqual(k_mean, H)

    def test_gaussian_curvature(self):
        """Validate evaluation of gaussian curvature"""
        uvpts = [[0,0],[1,0],[0,1],[1,1]]
        Ks = [
            flint.frac(1,3),
            flint.frac(1,3),
            flint(0),
            flint(0)
        ]
        for uv, K in zip(uvpts,Ks):
            k_g = self.tc.k_gaussian(*uv)
            self.assertTrue(abs(k_g-K) < 1.e-7)

    def test_principal_curvatures(self):
        """Validate the principle curvatures"""
        uvpts = [[0,0],[1,0],[0,1],[1,1]]
        kps = [
            [flint.frac(-1,3),flint(-1)],
            [flint.frac(-1,3),flint(-1)],
            [flint(0),flint(-1)],
            [flint(0),flint(-1)],
        ]
        for uv, kp in zip(uvpts, kps):
            kpc = self.tc.k_principal(*uv)
            self.assertTrue(abs(kpc[0]-kp[0]) < 1.e-7)
            self.assertTrue(abs(kpc[1]-kp[1]) < 1.e-7)
    
    def test_principal_curvature_directions(self):
        """Validate the principle curvatures and eigenvectors"""
        uvpts = [[0,0],[1,0],[0,1],[1,1]]
        kps = [
            [flint.frac(-1,3),flint(-1)],
            [flint.frac(-1,3),flint(-1)],
            [flint(0),flint(-1)],
            [flint(0),flint(-1)],
        ]
        evs = np.array([
            [[1,0],[0,-1]],
            [[1,0],[0,-1]],
            [[1,0],[0,-1]],
            [[1,0],[0,-1]],
        ], dtype=np.float64)
        for uv, kp, ev in zip(uvpts, kps, evs):
            kpc, evc = self.tc.k_princ_vec(*uv)
            self.assertTrue(abs(kpc[0]-kp[0]) < 1.e-7)
            self.assertTrue(abs(kpc[1]-kp[1]) < 1.e-7)
            self.assertTrue(np.allclose(ev,evc.astype(float)))
