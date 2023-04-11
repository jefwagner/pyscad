## @file types.py 
"""\
Defines Num, Vec, and Point as generic number, vector, and point types and test
for those types.
"""
# Copyright (c) 2023, Jef Wagner <jefwagner@gmail.com>
#
# This file is part of pyscad.
#
# pyscad is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pyscad is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pyscad. If not, see <https://www.gnu.org/licenses/>.

from typing import Union, Any, Sequence, Optional, Tuple
import numpy.typing as npt

import functools
import numpy as np
from flint import flint

np_numbers = (np.int8, np.uint8, np.int16, np.uint16, 
              np.int32, np.uint32, np.int64, np.uint64, 
              np.float32, np.float64)

Num = Union[float, flint]
Num.__doc__ = """Generic number as float or flint to be used for type hints"""
def is_num(x: Any) -> bool:
    """Test if input is a generic number
    @param x The input to test
    @return True if x is an integer, float, or flint
    """
    return isinstance(x, (int, float, *np_numbers, flint))

Vec = Sequence[Num]
Vec.__doc__ = """Generic vector as sequence of numbers to be used for type hints"""
def is_vec(v: Any, length: Optional[int] = None) -> bool:
    """Test if input is a generic vector
    @param v The input to test
    @param length [optional] The required length of the vector
    @return True if v is a sequence of numbers of appropriate length
    """
    try:
        a = (len(v) in [2,3]) if length is None else (len(v) == length)
        if not a:
            return False
        if isinstance(v, np.ndarray):
            return v.dtype in (*np_numbers, flint)
        return functools.reduce(lambda a, b: a and b, map(is_num, v))
    except:
        return False

Point = Union[Num, Vec]
Point.__doc__ = """Generic point as number or vector to be used for type hints"""
def is_point(a: Any) -> bool:
    """Test if input is a generic point
    @param a The input to test
    @return True if a is a number or 1, 2, or 3 length vector
    """
    return is_num(a) or is_vec(a) or is_vec(a, length=1)

# Simple Linear Algebra functions for 2x2 and 3x3 arrays
def det2(a: npt.NDArray) -> Num:
    """Calculate the determinant of a 2x2 matrix"""
    return a[0,0]*a[1,1] - a[0,1]*a[1,0]

def det3(a: npt.NDArray) -> Num:
    """Calculate the determinant of a 3x3 matrix"""
    return (a[0,0]*(a[1,1]*a[2,2] - a[1,2]*a[2,1]) +
            a[0,1]*(a[1,2]*a[2,0] - a[1,0]*a[2,2]) +
            a[0,2]*(a[1,0]*a[2,1] - a[1,1]*a[2,0]))

def det(a: npt.NDArray) -> Num:
    """Calculate the determinant of a 2x2 or 3x3 matrix"""
    if a.shape[0] == 2:
        return a[0,0]*a[1,1]-a[0,1]*a[0,1]
    elif a.shape[0] == 3:
        return det3(a)
    raise TypeError("Only works on 2 and 3 dimensional arrays")

def eig2(a: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """Calculate the eigenvalues and eigenvectors of a symetric 2x2 matrix"""
    if a[0,1] == 0:
        # Off diagonal is zero, so diagonal elements are the eigenvalues
        if a[0,0] >= a[1,1]:
            return np.diag(a), np.eye(2, dtype=flint)
        else:
            eigvals = np.array([a[1,1],a[0,0]])
            eigvecs = np.array([[0,1],[-1,0]], dtype=flint)
            return eigvals, eigvecs
    else:
        # Off diagonal is not zero, so need to find eigenvalues
        tra = a[0,0] + a[1,1]
        deta = a[0,0]*a[1,1]-a[0,1]*a[1,0]
        d = np.sqrt(tra*tra-4*deta)
        eigvals = np.array([0.5*(tra + d), 0.5*(tra - d)])
        if eigvals[0] == eigvals[1]:
            eigvecs = np.eye(2, dtype=flint)
        else:
            eigvecs = np.eye(2, dtype=flint)
            eigvecs[0] = (a-eigvals[1]*np.eye(2, dtype=flint))[:,0]
            if np.alltrue(eigvecs[1] == np.zeros(2, dtype=flint)):
                eigvecs[0] = (a-eigvals[0]*np.eye(2, dtype=flint))[:,1]
            if eigvecs[0,0] > 0 or (eigvecs[0,0] == 0 and eigvecs[0,1] > 0):
                norm = np.sqrt(np.sum(eigvecs[0]*eigvecs[0]))
            else:
                norm = -np.sqrt(np.sum(eigvecs[0]*eigvecs[0]))
            eigvecs[0] /= np.sqrt(np.sum(eigvecs[0]*eigvecs[0]))
            eigvecs[1,0] = -eigvecs[0,1]
            eigvecs[1,1] = eigvecs[0,0]
        return eigvals, eigvecs

def eig3(a: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """Calculate the eigenvalues and eigenvectors of a symetric 3x3 matrix"""
    p1 = a[0,1]*a[0,1] + a[0,2]*a[0,2] + a[1,2]*a[1,2]
    if p1 == 0:
        # if off-diagonal are zero, then the diagonal is the eigenvalues
        # and the vectors columns of the identify matrix
        l = np.diag(a).copy()
        v = np.eye(3, dtype=flint)
        if l[1] < l[2]:
            l[1], l[2] = l[2], l[1]
            v[1], v[2] = v[2], -v[1]
        if l[0] < l[1]:
            l[0], l[1] = l[1], l[0]
            v[0], v[1] = v[1], -v[0]
        if l[1] < l[2]:
            l[1], l[2] = l[2], l[1]
            v[1], v[2] = v[2], -v[1]
        return l, v
    else:
        tra = a[0,0] + a[1,1] + a[2,2]
        q = tra/3
        p2 = ((a[0,0]-q)*(a[0,0]-q) + 
              (a[1,1]-q)*(a[1,1]-q) + 
              (a[2,2]-q)*(a[2,2]-q)) + 2*p1
        p = np.sqrt(p2/6)
        b = (a-q*np.eye(3))/p
        phi = np.arccos(det3(b)/2)/3
        eig0 = q + 2*p*np.cos(phi)
        eig2 = q + 2*p*np.cos(phi + 2*np.pi/3)
        eig1 = tra - eig0 - eig2
        eigvals = np.array([eig0, eig1, eig2])
        # special cases for repeated eigenvalues
        eigvecs = np.eye(3, dtype=flint)
        if (eig0 == eig1) and (eig0 == eig2):
            # if all 3 eigenvalues the same, just use the identity
            pass
        else:
            # Use Cayley-Hamilton theory to get two eigenvectors
            m0 = (a-eig0*np.eye(3, dtype=flint))
            m1 = (a-eig1*np.eye(3, dtype=flint))
            m2 = (a-eig2*np.eye(3, dtype=flint))
            eigvecs[0] = (m1.dot(m2))[:,0]
            if np.alltrue( eigvecs[0] == np.zeros(3) ):
                eigvecs[0] = (m1.dot(m2))[:,1]
                if np.alltrue( eigvecs[0] == np.zeros(3) ):
                    eigvecs[0] = (m1.dot(m2))[:,2]
            eigvecs[2] = (m0.dot(m1))[:,0]
            if np.alltrue( eigvecs[2] == np.zeros(3) ):
                eigvecs[2] = (m0.dot(m1))[:,1]
                if np.alltrue( eigvecs[2] == np.zeros(3) ):
                    eigvecs[2] = (m0.dot(m1))[:,2]
            # Normalize the first two vectors and orient in a well defined something
            if (eigvecs[0,0] > 0 or
                (eigvecs[0,0] == 0 and eigvecs[0,1] > 0) or
                (eigvecs[0,0] == 0 and eigvecs[0,1] == 0 and eigvecs[0,2] > 0)):
                norm = np.sqrt(np.sum(eigvecs[0]*eigvecs[0]))
            else:
                norm = -np.sqrt(np.sum(eigvecs[0]*eigvecs[0]))
            eigvecs[0] /= norm
            if (eigvecs[2,0] > 0 or
                (eigvecs[2,0] == 0 and eigvecs[2,1] > 0) or
                (eigvecs[2,0] == 0 and eigvecs[2,1] == 0 and eigvecs[2,2] > 0)):
                norm = np.sqrt(np.sum(eigvecs[2]*eigvecs[2]))
            else:
                norm = -np.sqrt(np.sum(eigvecs[2]*eigvecs[2]))
            eigvecs[0] /= norm
            # Use cross product to get last eigenvector
            eigvecs[1] = np.cross(eigvecs[2], eigvecs[0])
        return eigvals, eigvecs


# def svd(a: npt:NDArray) -> Num:
#     ata = np.matmul(a.T, a)


# Convenience functions for turning a flint/number or array into a JSON
# serializable format.
def num_json(x: Num) -> Union[float, dict]:
    """Convert a generic number into a JSON serializable object"""
    if isinstance(x, flint):
        return {
            'type': 'flint', 
            'a': float(x.a), 
            'b': float(x.b), 
            'v': float(x.v)
        }
    elif isinstance(x, (int, float, *np_numbers)):
        return x
    else:
        raise TypeError('Can only convert general number type')

def array_json(a: npt.NDArray[Num]) -> list:
    """Convert a numpy array into a JSON serializable object"""
    if not isinstance(a, np.ndarray) or a.dtype not in (*np_numbers, flint):
        raise TypeError('Can only convert NumPy ndarrays of numbers')
    if len(a.shape) == 1:
        return [num_json(x) for x in a]
    else:
        return [array_json(suba) for suba in a]
        
