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

import numpy as np
import numpy.typing as npt
from flint import flint

from ._c_affine import eye, from_mat 
from ._c_affine import trans, scale, rot, refl, skew
from ._c_affine import rescale, apply_vert

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
    return np.matmul(lhs, rhs)

def transform_reduce(trasforms: list[npt.NDArray[flint]]) -> npt.NDArray[flint]:
    """Reduce a sequence of affine transforms into a single affine transform.

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
        np.matmul(tr, out, out=out)
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
    if v_in.shape[0] not in [3,4]:
        raise ValueError('foo')
    v_out = np.empty(v_in.shape, dtype=flint)
    if v_in.shape[0] == 3:
        # apply for 3-length vertices
        apply_vert(transform, v_in, v_out)
    else:
        # apply for 4-length homogenous coordinates
        v_out = np.inner(transform, v_in)
        rescale(v_out, v_out)
    return v_out
