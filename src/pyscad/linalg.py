## @file linalg.py 
"""\
Define some linear algebra functions for two and three dimensional matricies
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

import numpy.typing as npt
import numpy as np
from flint import flint

from .types import *

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
        return a[0,0]*a[1,1] - a[0,1]*a[1,0]
    elif a.shape[0] == 3:
        return det3(a)
    raise TypeError("Only works on 2 and 3 dimensional arrays")

def eig2(a: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """Calculate the eigenvalues and eigenvectors of a symmetric 2x2 matrix"""
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
    """Calculate the eigenvalues and eigenvectors of a symmetric 3x3 matrix"""
    p1 = a[0,1]*a[0,1] + a[0,2]*a[0,2] + a[1,2]*a[1,2]
    if p1 == 0:
        # if off-diagonal are zero, then the diagonal is the eigenvalues
        # and the vectors columns of the identify matrix
        l = np.diag(a).copy()
        v = np.eye(3, dtype=flint)
        # Sort the eigenvectors and eigenvalues
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
        # else use Viete's trig solution to the cubic
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
        # calculate the vectors
        eigvecs = find_vecs3(a, eigvals)
        # eigvecs = refine_vec3(a, eigvecs)
        return eigvals, eigvecs

def find_vecs3(a: npt.NDArray, eigvals: npt.NDArray) -> npt.NDArray:
    """Find the eigenvectors for a 3x3 system"""
    eig0, eig1, eig2 = eigvals
    eigvecs = np.eye(3, dtype=flint)
    if (eig0 == eig1) and (eig0 == eig2):
        # no sorting if all 3 are equal
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
        # Normalize and orient the first eigenvector
        if (eigvecs[0,0] > 0 or
            (eigvecs[0,0] == 0 and eigvecs[0,1] > 0) or
            (eigvecs[0,0] == 0 and eigvecs[0,1] == 0 and eigvecs[0,2] > 0)):
            norm = np.sqrt(np.sum(eigvecs[0]*eigvecs[0]))
        else:
            norm = -np.sqrt(np.sum(eigvecs[0]*eigvecs[0]))
        eigvecs[0] /= norm
        # Make sure last eigenvector is orthogonal
        eigvecs[2] -= eigvecs[2].dot(eigvecs[0])*eigvecs[0]
        # Then normalized and orient last eigenvector
        if (eigvecs[2,0] > 0 or
            (eigvecs[2,0] == 0 and eigvecs[2,1] > 0) or
            (eigvecs[2,0] == 0 and eigvecs[2,1] == 0 and eigvecs[2,2] > 0)):
            norm = np.sqrt(np.sum(eigvecs[2]*eigvecs[2]))
        else:
            norm = -np.sqrt(np.sum(eigvecs[2]*eigvecs[2]))
        eigvecs[2] /= norm
        # Use cross product to get last eigenvector
        eigvecs[1] = np.cross(eigvecs[2], eigvecs[0])
    return eigvecs

def refine_vec3(a: npt.NDArray, q: npt.NDArray) -> npt.NDArray:
    """Refine the eigenvectors"""
    q = q.T.copy()
    l = np.zeros((3,), dtype=flint)
    e = np.zeros((3,3), dtype=flint)
    r = np.eye(3) - (q.T).dot(q)
    s = (q.T).dot(a).dot(q)
    for i in range(3):
        l[i] = s[i,i]/(1-r[i,i])
    for i in range(3):
        for j in range(3):
            e[i,j] = 0.5*r[i,j] if l[i] == l[j] else (s[i,j]+l[j]*r[i,j])/(l[j]-l[i])
    q += q.dot(e)
    q = q.T.copy()
    for i, v in enumerate(q):
        q[i] /= np.sqrt(np.sum(v*v))
    return q

def eig(a: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """Calculate eigenvalues and eigenvectors for a symmetric 2x2 or 3x3 matrix"""
    if a.shape[0] == 2:
        return eig2(a)
    elif a.shape[0] == 3:
        return eig3(a)
    raise TypeError("Only works on 2 and 3 dimensional arrays")

def svd(a: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Calculate the singular value decomposition for a 2x2 or 3x3 matrix
    @param a The input matrix
    @return A tuple of 3 numpy arrays (u, sig, vt)
    u, and vt are a unitary matricies and sig is a vector of the singular\
    values. The input matrix can be identified as a = u.diag(sig).vt
    """
    b = (a.T).dot(a)
    l, vt = eig(b)
    sig = np.sqrt(l)
    u = a.dot(vt.T).dot(np.diag(1/sig))
    return u, sig, vt
