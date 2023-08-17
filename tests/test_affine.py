import numpy as np
from flint import flint

from pyscad import AffineTransform

import pytest

class TestCreation:

    def test_init(self):
        a = AffineTransform()
        assert isinstance(a, AffineTransform)
        assert str(a) == 'AffineTransform'
        assert np.all( a.array == np.eye(4))

    def test_from_mat_exc(self):
        with pytest.raises(ValueError):
            a = AffineTransform.from_mat()
        with pytest.raises(ValueError):
            a = AffineTransform.from_mat(1, 1)
        with pytest.raises(ValueError):
            a = AffineTransform.from_mat([1,2,3])
        with pytest.raises(ValueError):
            a = AffineTransform.from_mat([[1,2,3],[1,2,3]])
        with pytest.raises(ValueError):
            a = AffineTransform.from_mat([[[1,2,3],[1,2,3]]])

    def test_from_mat_4x4(self):
        a = AffineTransform.from_mat([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
        assert isinstance(a, AffineTransform)
        assert str(a) == 'AffineTransform'
        assert np.all( a.array == np.arange(16).reshape((4,4)))

    def test_from_mat_4x3(self):
        a = AffineTransform.from_mat([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
        b = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[0,0,0,1]])
        assert isinstance(a, AffineTransform)
        assert str(a) == 'AffineTransform'
        assert np.all( a.array == b)

    def test_from_mat_3x3(self):
        a = AffineTransform.from_mat([[0,1,2],[3,4,5],[6,7,8]])
        b = np.array([[0,1,2,0],[3,4,5,0],[6,7,8,0],[0,0,0,1]])
        assert isinstance(a, AffineTransform)
        assert str(a) == 'AffineTransform'
        assert np.all( a.array == b)


class TestTranslation:

    def test_translation_exc(self):
        with pytest.raises(ValueError):
            a = AffineTransform.Translation()
        with pytest.raises(ValueError):
            a = AffineTransform.Translation(1,2)
        with pytest.raises(ValueError):
            a = AffineTransform.Translation([1,2])
        with pytest.raises(ValueError):
            a = AffineTransform.Translation([[1,2,3]])

    def test_translation(self):
        a = AffineTransform.Translation([1,2,3])
        assert isinstance(a, AffineTransform)
        assert str(a) == 'Translation'
        b = np.array([[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]])
        assert np.all( a.array == b )

    def test_translation(self):
        a = AffineTransform.Translation([5,6,7], center=[0,1,3])
        assert isinstance(a, AffineTransform)
        assert str(a) == 'Translation'
        b = np.array([[1,0,0,5],[0,1,0,6],[0,0,1,7],[0,0,0,1]])
        assert np.all( a.array == b )


class TestScale:

    def test_scale_exc(self):
        with pytest.raises(ValueError):
            a = AffineTransform.Scale()
        with pytest.raises(ValueError):
            a = AffineTransform.Scale(1,2)
        with pytest.raises(ValueError):
            a = AffineTransform.Scale([1])
        with pytest.raises(ValueError):
            a = AffineTransform.Scale([[1,2,3]])

    def test_scale_scalar(self):
        comp = np.eye(4, dtype=np.float64)
        a = AffineTransform.Scale(2)
        for i in range(3):
            comp[i,i] = 2
        assert str(a) == 'Scale'
        assert np.all( a.array == comp )
        a = AffineTransform.Scale(0.5)
        for i in range(3):
            comp[i,i] = 0.5
        assert str(a) == 'Scale'
        assert np.all( a.array == comp )
        a = AffineTransform.Scale(flint(1.0)/3)
        for i in range(3):
            comp[i,i] = 1/3
        assert str(a) == 'Scale'
        assert np.all( a.array == comp )

    def test_scale_vec(self):
        comp = np.eye(4, dtype=np.float64)
        a = AffineTransform.Scale([2,3,4])
        comp[0,0] = 2
        comp[1,1] = 3
        comp[2,2] = 4
        assert str(a) == 'Scale'
        assert np.all( a.array == comp )


class TestRotation:

    def test_rotation_exc(self):
        with pytest.raises(ValueError):
            a = AffineTransform.Rotation()
        with pytest.raises(ValueError):
            a = AffineTransform.Rotation(1)
        with pytest.raises(ValueError):
            a = AffineTransform.Rotation(1,2,3)
        with pytest.raises(ValueError):
            a = AffineTransform.Rotation(1,1)
        with pytest.raises(ValueError):
            a = AffineTransform.Rotation('xoo',1)
        with pytest.raises(ValueError):
            a = AffineTransform.Rotation([1],1)

    def test_rot_x(self):
        a = AffineTransform.Rotation('x', np.pi/2)
        assert isinstance(a, AffineTransform)
        assert str(a) == 'Rotation'
        b = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
        assert np.all( a.array == b )

    def test_rot_y(self):
        a = AffineTransform.Rotation('Y', np.pi/2)
        assert isinstance(a, AffineTransform)
        assert str(a) == 'Rotation'
        b = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        assert np.all( a.array == b )

    def test_rot_z(self):
        a = AffineTransform.Rotation('z', np.pi/2)
        assert isinstance(a, AffineTransform)
        assert str(a) == 'Rotation'
        b = np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
        assert np.all( a.array == b )

    def test_rot_aa(self):
        a = AffineTransform.Rotation('x', 1)
        b = AffineTransform.Rotation([1,0,0], 1)
        assert np.all( a.array == b.array )
        a = AffineTransform.Rotation('y', 1.5)
        b = AffineTransform.Rotation([0,2,0], 1.5)
        assert np.all( a.array == b.array )
        a = AffineTransform.Rotation('z', 0.5)
        b = AffineTransform.Rotation([0,0,0.5], 0.5)
        assert np.all( a.array == b.array )


class TestRefection:

    def test_refl_exc(self):
        with pytest.raises(ValueError):
            a = AffineTransform.Reflection()

class TestSkew:

    def test_skew_exc(self):
        with pytest.raises(ValueError):
            a = AffineTransform.Skew()