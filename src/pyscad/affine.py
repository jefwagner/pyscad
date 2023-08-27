## @file affine.py 
"""
"""
# Copyright (c) 2023, Jef Wagner <jefwagner@gmail.com>
#
# This file is part of pyscad.
#
# pyscad is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pyscad is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pyscad. If not, see <https://www.gnu.org/licenses/>.
from typing import Union, Optional

import numpy as np
import numpy.typing as npt
from flint import flint

from ._c_affine import rescale, apply_vert

NumLike = Union[int, float, flint]

def eye(dtype: npt.DTypeLike = flint) -> npt.NDArray:
    """Create an identify affine transform"""
    return np.eye(4, dtype=dtype)

def from_mat(mat: npt.ArrayLike, dtype: npt.DTypeLike = flint) -> npt.NDArray:
    """Create a new generic affine transform from a 4x4, 3x4 or 3x3 matrix
    
    * A 3x3 matrix will only specify the linear transformation.
    * A 3x4 matrix will specify the linear transformation and translation.
    * A 4x4 will specify the linear transformation, translation, and perspective
        transformation.
    
    :param mat: The input matrix (any properly shaped nested sequence type).
    
    :return: An AffineTransform object corresponding to the matrix");
    """
    mat = np.array(mat, dtype=dtype)
    if mat.shape not in ((4,4),(3,4),(3,3)):
        raise ValueError('Argument must be a 4x4, 3x4, or 3x3 array')
    trans = eye(dtype=dtype)
    I, J = mat.shape
    for i in range(I):
        for j in range(J):
            trans[i,j] = mat[i,j]
    return trans

def _relocate_center(trans: npt.NDArray, center: npt.ArrayLike):
    """Relocate the center of a linear transformation"""
    center = np.array(center)
    if center.shape != (3,):
        raise ValueError('The center should be a 3 length [cx, cy, cz]')
    a = trans[:3,:3]
    d = center - a.dot(center)
    for i in range(3):
        trans[i,3] = d[i]

def trans(d: npt.ArrayLike, 
          center: Optional[npt.NDArray] = None, 
          dtype: npt.DTypeLike = flint) -> npt.NDArray:
    """Create a new pure translation transformation.
    
    :param d: A 3-length sequence [dx, dy, dz]
    :param center: Ignored
    
    :return: An pure translation AffineTransformation.");
    """
    d = np.array(d, dtype=dtype)
    if d.shape != (3,):
        raise ValueError('The translation argument `d` should be a 3 length [dx, dy, dz]')
    trans = eye(dtype=dtype)
    for i in range(3):
        trans[i,3] = d[i]
    return trans

def scale(s: Union[NumLike, npt.ArrayLike], 
          center: Optional[npt.NDArray] = None, 
          dtype: npt.DTypeLike = flint) -> npt.NDArray:
    """Create a new pure scaling transformation.
    
    :param s: A scalar or 3-length sequence [sx, sy, sz] for scaling along each n
    :param center: Optional 3-length center position [cx, cy, cz] for the scaling
        transform
    
    :return: A scaling if AffineTransformation."""
    s = np.array(s, dtype=dtype)
    # Scalar input
    if s.shape == ():
        s = np.array([s,s,s], dtype=flint)
    if s.shape != (3,):
        raise ValueError('The scale argument `s` must be a scalar or a 3 length [sx, sy, sz]')
    trans = eye(dtype=dtype)
    for i in range(3):
        trans[i,i] = s[i]
    if center is not None:
        _relocate_center(trans, center)
    return trans

def rot(axis: Union[str, npt.ArrayLike], 
        angle: NumLike,
        center: Optional[npt.ArrayLike] = None,
        dtype: npt.DTypeLike = flint) -> npt.NDArray:
    """Create a new pure rotation transformation.
    
    :param axis: The character 'x','y','z' or a three length vector [ax, ay, az]
    :param angle: The angle in radians to rotate
    :param center: Optional 3-length position [cx, cy, cz] for to specify a point
        on the axix of rotation
    
    :return: A rotation AffineTransformation."""
    if isinstance(axis, str):
        if len(axis) != 1 or axis.lower()[0] not in ['x','y','z']:
            raise ValueError("Axis must be either he character 'x','y','z' or a three length vector [ax, ay, az]")
        axis = axis.lower()[0]
    else:
        axis = np.array(axis, dtype=dtype)
        if axis.shape != (3,):
            raise ValueError("Axis must be either he character 'x','y','z' or a three length vector [ax, ay, az]")
        axis_len = np.sqrt(np.sum(np.dot(axis, axis)))
        if axis_len != 1:
            axis = axis/axis_len
    angle = np.array(angle, dtype=dtype)
    s, c = np.sin(angle), np.cos(angle)
    trans = eye(dtype=dtype)
    if isinstance(axis, str):
        if axis == 'x':
            trans[1,1] = c
            trans[2,2] = c
            trans[1,2] = -s
            trans[2,1] = s
        elif axis == 'y':
            trans[0,0] = c
            trans[2,2] = c
            trans[0,2] = s
            trans[2,0] = -s
        elif axis == 'z':
            trans[0,0] = c
            trans[1,1] = c
            trans[0,1] = -s
            trans[1,0] = s
    else:
        for i in range(3):
            for j in range(3):
                trans[i,j] =axis[i]*axis[j]*(1-c)
            trans[i,i] += c
        trans[0,1] -= axis[2]*s
        trans[1,0] += axis[2]*s
        trans[0,2] += axis[1]*s
        trans[2,0] -= axis[1]*s
        trans[1,2] -= axis[0]*s
        trans[2,1] += axis[0]*s
    if center is not None:
        _relocate_center(trans, center)
    return trans

def refl(n: Union[str, npt.ArrayLike],
         center: Optional[npt.ArrayLike] = None,
         dtype: npt.DTypeLike = flint) -> npt.NDArray:
    """Create a new pure reflection transformation.
    
    :param normal: The character 'x','y','z' or a 3 length [ux, uy, uz] vector for
        the normal vector for the reflection plane.
    :param center: Optional 3-length center position [cx, cy, cz] a point on the
        plane of reflection operation.
    
    :return: A skew AffineTransformation.");
    """
    if isinstance(n, str):
        if len(n) != 1 or n.lower()[0] not in ['x','y','z']:
            raise ValueError("n must be either the character 'x','y','z' or a three length vector [ax, ay, az]")
        n = n.lower()[0]
    else:
        n = np.array(n, dtype=dtype)
        if n.shape != (3,):
            raise ValueError("n must be either he character 'x','y','z' or a three length vector [ax, ay, az]")
        n_len = np.sqrt(np.sum(np.dot(n, n)))
        if n_len != 1:
            n = n/n_len
    trans = eye(dtype=dtype)
    if isinstance(n, str):
        if n == 'x':
            trans[0,0] = -1
        elif n == 'y':
            trans[1,1] = -1
        elif n == 'z':
            trans[2,2] = -1
    else:
        for i in range(3):
            for j in range(3):
                trans[i,j] -= 2*n[i]*n[j]
    if center is not None:
        _relocate_center(trans, center)
    return trans    

def skew(n: Union[str, npt.ArrayLike],
         s: npt.ArrayLike,
         center: Optional[npt.ArrayLike] = None,
         dtype: npt.DTypeLike = flint) -> npt.NDArray:
    """Create a new pure skew transformation.
    
    :param n: The character 'x','y','z' or a 3 length [nx, ny, nz] normal
        vector to define the skew (shear) plane.
    :param s: A 3 length [sx, sy, sz] vector for the skew direction.
    :param center: Optional 3-length center position [cx, cy, cz] for the center of
        the skew operation.
    
    :return: A skew AffineTransformation."""
    if isinstance(n, str):
        if len(n) != 1 or n.lower()[0] not in ['x','y','z']:
            raise ValueError("n must be either the character 'x','y','z' or a three length vector [ax, ay, az]")
        n = n.lower()[0]
        n_len = 1
        if n == 'x':
            n = np.array([1,0,0], dtype=dtype)
        elif n == 'y':
            n = np.array([0,1,0], dtype=dtype)
        elif n == 'z':
            n = np.array([0,0,1], dtype=dtype)
    else:
        n = np.array(n, dtype=dtype)
        if n.shape != (3,):
            raise ValueError("n must be either he character 'x','y','z' or a three length vector [ax, ay, az]")
        n_len = np.sqrt(np.sum(np.dot(n, n)))
        if n_len != 1:
            n = n/n_len
    s = np.array(s, dtype=dtype)
    if s.shape != (3,):
        raise ValueError("s must be a three length vector [sx, sy, sz]")
    # Get the part of s perpendicular to n
    s_dot = s.dot(n)
    s -= s_dot*n
    trans = eye(dtype=dtype)
    for i in range(3):
        for j in range(3):
            trans[i,j] += s[i]*n[j]
    if center is not None:
        _relocate_center(trans, center)
    return trans

def combine(lhs: npt.NDArray[flint], rhs: npt.NDArray[flint]) -> npt.NDArray[flint]:
    """Combine two affine transforms into a single transform. 

    This is simply the matrix multiplication of the two transforms, and so the
        order of the two transforms matters. The resulting transform is the same
        as applying the right-hand-side transform first, then the
        left-hand-side.
    
    :param lhs: The left-hand-side affine transform
    :param rhs: The right-hand-side affine transform

    :return: The resulting combined transform
    """
    return np.dot(lhs, rhs)

def transform_reduce(transforms: list[npt.NDArray[flint]]) -> npt.NDArray[flint]:
    r"""Reduce a sequence of affine transforms into a single affine transform.

    This is the same as a repeated matrix multiplication, and so order of the
        transforms matter. The result is the same as the first transform applied
        followed by the second, and so on. A transform list `[T0, T1, T2, ...,
        TN]` would reduce to

    $T_{\text{reduced}} = T_N\cdot T_{N-1] \cdot \ldots \cdot T1 \cdot T0.$
    
    :param transforms: The sequence of affine transforms

    :return: The resulting reduced affine transform
    """
    out = eye()
    for tr in transforms:
        np.dot(tr, out, out=out)
    return out

def apply(transform: npt.NDArray[flint], v_in: npt.ArrayLike) -> npt.NDArray[flint]:
    """Apply a transform to a single vertex or array of vertices.

    The vertex can either be a 3-length coordinates [x,y,z] or 4-length 
        homogeneous coordinates [x,y,z,1]. For a 3-length vertex the result
        is the same as it would be for the same homogenous coordinate.
    
    :param transform: The affine transform to apply
    :param v_in: The vertex or array of vertices

    :return: A new transformed vertex or array of transformed vertices
    """
    if not isinstance(v_in, np.ndarray):
        v_in = np.array(v_in)
    if len(v_in.shape) == 0:
        raise TypeError('foo')
    if v_in.shape[-1] not in [3,4]:
        raise ValueError('foo')
    v_out = np.empty(v_in.shape, dtype=flint)
    if v_in.shape[-1] == 3:
        # apply for 3-length vertices
        apply_vert(transform, v_in, v_out)
    else:
        # apply for 4-length homogenous coordinates
        v_out = np.inner(v_in, transform)
        rescale(v_out, v_out)
    return v_out
