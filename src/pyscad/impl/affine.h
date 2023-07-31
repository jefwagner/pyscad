/// @file affine.h Affine transforms
//
// Copyright (c) 2023, Jef Wagner <jefwagner@gmail.com>
//
// This file is part of pyscad.
//
// pyscad is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//
// pyscad is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// pyscad. If not, see <https://www.gnu.org/licenses/>.
//

#ifndef PYSCAD_AFFINE_H
#define PYSCAD_AFFINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>

#include <flint.h>

typdef enum {
    AT_GENERIC,
    AT_TRANSLATION,
    AT_ROTATION,
    AT_SCALE,
    AT_REFLECT,
    AT_SKEW,
} TransformType;

/// @brief An affine transform
/// @param array The 4x4 array of c flint objects
typedef struct {
    PyObject_HEAD
    TransformType type;
    flint array[16];    
} PyAffineTrans;

/// @brief The flint PyTypeObject
static PyTypeObject PyAffineTrans_Type;

//------------------------------------------------------------
// Original module includes
//------------------------------------------------------------
#ifdef PYSCAD_AFFINE_MODULE


//------------------------------------------------------------
// Capsule API includes
//------------------------------------------------------------
#else // PYSCAD_AFFINE MODULE


#endif //PYSCAD_AFFINE_MODULE

/// @brief Check if an object is an affine transform
/// @param ob The PyObject to check
/// @return 1 if the object is an affine transform, 0 otherwise
static inline int PyAffineTrans_Check(PyObject* ob) {
    return PyObject_IsInstance(ob, (PyObject*) &PyAffineTrans_Type);
}


#ifdef __cplusplus
}
#endif

#endif // PYSCAD_AFFINE_H