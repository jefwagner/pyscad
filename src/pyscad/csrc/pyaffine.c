/// @file pyaffine.h 
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

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include <flint.h>
#include <numpy_flint.h>

#define AFFINE_MODULE
#include "affine.h"
// #include "pyaffine.h"

#define NEW_AFFINE(NP_ARR, FLINT_ARR) \
PyArrayObject* NP_ARR = NULL;\
flint* FLINT_ARR = NULL;\
const npy_intp _shape[] = {4,4};\
NP_ARR = (PyArrayObject*) PyArray_SimpleNew(2, _shape, NPY_FLINT);\
if (NP_ARR == NULL) {\
    PyErr_SetString(PyExc_SystemError, "Error allocating new affine transform array");\
    return NULL;\
}\
FLINT_ARR = PyArray_DATA(NP_ARR)

/// @brief Create an identity affine transform
PyDoc_STRVAR( eye_docstring, "Create an identify affine transform");
static PyObject* pyaffine_eye(PyObject* self, PyObject* args) {
    NEW_AFFINE(aff, arr);
    affine_eye(arr);
    return (PyObject*) aff;
}

/// @brief Classmethod for making an AffineTransform from a matrix
PyDoc_STRVAR( from_mat_docstring, "\
Create a new generic affine transform from a 4x4, 3x4 or 3x3 matrix\n\
\n\
* A 3x3 matrix will only specify the linear transformation.\n\
* A 3x4 matrix will specify the linear transformation and translation.\n\
* A 4x4 will specify the linear transformation, translation, and perspective\n\
    transformation.\n\
\n\
:param mat: The input matrix (any properly shaped nested sequence type).\n\
\n\
:return: An AffineTransform object corresponding to the matrix");
static PyObject* pyaffine_from_mat(PyObject* self, PyObject* arg) {
    int i, j;
    PyArrayObject* in_arr = NULL;
    npy_intp* in_shape = NULL;
    NEW_AFFINE(out_arr, out_data);    
    in_arr = (PyArrayObject*) PyArray_FROM_OT(arg, NPY_FLINT);
    if (in_arr == NULL) {
        PyErr_SetString(PyExc_TypeError, "Argument must castable to a numpy array");
        Py_DECREF(out_arr);
        return NULL;
    }
    Py_INCREF(in_arr);
    in_shape = PyArray_SHAPE(in_arr);
    if (
            PyArray_NDIM(in_arr) != 2 ||
            !((in_shape[0] == 4 && in_shape[1] == 4) ||
              (in_shape[0] == 3 && in_shape[1] == 4) ||             
              (in_shape[0] == 3 && in_shape[1] == 3))             
        ) {
        PyErr_SetString(PyExc_ValueError, "Argument must be a 4x4, 3x4, or 3x3 array");
        Py_DECREF(in_arr);
        Py_DECREF(out_arr);
        return NULL;
    }
    for (i=0; i<in_shape[0]; i++) {
        for (j=0; j<in_shape[1]; j++) {
            out_data[4*i+j] = *((flint*) PyArray_GETPTR2(in_arr, i, j));
        }
    }
    if (in_shape[0] == 3) {
        for (j=0; j<3; j++) {
            out_data[4*3+j] = int_to_flint(0);
        }
        out_data[4*3+3] = int_to_flint(1);
    }
    if (in_shape[1] == 3) {
        for(i=0; i<3; i++) {
            out_data[4*i+3] = int_to_flint(0);
        }
    }
    return (PyObject*) out_arr;
}

/// @brief Create a new pure translation AffineTransform
PyDoc_STRVAR( translation_docstring, "\
Create a new pure translation transformation.\n\
\n\
:param d: A 3-length sequence [dx, dy, dz]\n\
:param center: Ignored\n\
\n\
:return: An pure translation AffineTransformation.");
static PyObject* pyaffine_translation(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* translation_keywords[] = {"d", "center", NULL};
    PyObject* d_obj = NULL;
    PyArrayObject* d_arr = NULL;
    bool valid_d = false;
    PyObject* c_obj = NULL;
    NEW_AFFINE(out_arr, out_data);

    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O", translation_keywords, &d_obj, &c_obj)) {
        Py_INCREF(d_obj);
        d_arr = (PyArrayObject*) PyArray_FROM_OT(d_obj, NPY_FLINT);
        if (d_arr != NULL) {
            Py_INCREF(d_arr);
            if (PyArray_NDIM(d_arr) == 1 && PyArray_SHAPE(d_arr)[0] == 3) {
                affine_set_translation(out_data, (flint*) PyArray_DATA(d_arr));
                valid_d = true;
            }
        }
        Py_DECREF(d_obj);
    }
    if (!valid_d) {
        PyErr_SetString(PyExc_ValueError, "Translation argument should be a 3 length sequence");
        Py_DECREF(out_arr);
        return NULL;
    }
    return (PyObject*) out_arr;
}

// Temporarily keeping around to remind myself how to 'print debug'
// printf("%s\n", Py_TYPE(s_arg)->tp_name);
// printf("%d\n", PyFlint_Check(s_arg));
/// @brief Create a new pure scaling AffineTransform
PyDoc_STRVAR( scale_docstring, "\
Create a new pure scaling transformation.\n\
\n\
:param s: A scalar or 3-length sequence [sx, sy, sz]\n\
:param center: Optional 3-length center position [cx, cy, cz] for the scaling\n\
    transform\n\
\n\
:return: A scaling if AffineTransformation.");
static PyObject* pyaffine_scale(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* scale_keywords[] = {"s", "center", NULL};
    int i;
    long long n;
    double d;
    // variable for scale
    bool valid_scale = false;
    PyObject* s_arg = NULL;
    PyArrayObject* s_arr = NULL;
    flint s[3];
    // variable for center
    bool use_center = false;
    PyObject* c_arg = NULL;
    PyArrayObject* c_arr = NULL;
    flint c[3];
    // allocate new affine transform
    NEW_AFFINE(out_arr, out_data);

    // Parse args
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O", scale_keywords, &s_arg, &c_arg)) {
        // Check for int, double, of flint scalar
        // Parse out the scale argument
        Py_INCREF(s_arg);
        if (PyLong_Check(s_arg)) {
            n = PyLong_AsLongLong(s_arg);
            for (i=0; i<3; i++) {
                s[i] = int_to_flint(n);
            }
            valid_scale = true;
        }
        else if (PyFloat_Check(s_arg)) {
            d = PyFloat_AsDouble(s_arg);
            for (i=0; i<3; i++) {
                s[i] = double_to_flint(d);
            }
            valid_scale = true;
        }
        else if (PyFlint_Check(s_arg)) {
            for (i=0; i<3; i++) {
                s[i] = PyFlint_AsFlint(s_arg);
            }
            valid_scale = true;
        }
        else {
            s_arr = (PyArrayObject*) PyArray_FROM_OT(s_arg, NPY_FLINT);
            if (s_arr != NULL) {
                if (PyArray_NDIM(s_arr) == 1) {
                    if (PyArray_SHAPE(s_arr)[0] == 3) {
                        for (i=0; i<3; i++) {
                            s[i] = *((flint*) PyArray_GETPTR1(s_arr, i));
                        }
                        valid_scale = true;
                    }
                }
                Py_DECREF(s_arr);
            }
        }
        Py_DECREF(s_arg);
        // Parse out the center argument
        if (c_arg != NULL) {
            Py_INCREF(c_arg);
            c_arr = (PyArrayObject*) PyArray_FROM_OT(c_arg, NPY_FLINT);
            if (c_arr != NULL) {
                if (PyArray_NDIM(c_arr) == 1) {
                    if (PyArray_SHAPE(c_arr)[0] == 3) {
                        for (i=0; i<3; i++) {
                            c[i] = *((flint*) PyArray_GETPTR1(c_arr, i));
                        }
                        use_center = true;
                    }
                }
                Py_DECREF(c_arr);
            }
            if (!use_center) {
                PyErr_SetString(PyExc_ValueError, "center must be a 3-length position [cx, cy, cz]");
                Py_DECREF(out_arr);
                Py_DECREF(c_arg);
                return NULL;
            }
            Py_DECREF(c_arg);
        }
    }
    if (!valid_scale) {
        PyErr_SetString(PyExc_ValueError, "s must be a scalar or scalar or 3-length non-uniform scaling [sx, sy, sz]");
        Py_DECREF(out_arr);
        return NULL;
    }
    affine_set_scale(out_data, s);
    if (use_center) {
        affine_relocate_center(out_data, c);
    }
    return (PyObject*) out_arr;
}

/// @brief Create a new pure rotation AffineTransform
PyDoc_STRVAR( rotation_docstring, "\
Create a new pure rotation transformation.\n\
\n\
:param axis: The character 'x','y','z' or a three length vector [ax, ay, az]\n\
:param angle: The angle in radians to rotate\n\
:param center: Optional 3-length position [cx, cy, cz] for to specify a point\n\
    on the axix of rotation\n\
\n\
:return: A rotation AffineTransformation.");
static PyObject* pyaffine_rotation(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* rotation_keywords[] = {"axis", "angle", "center", NULL};
    int i;
    // axis variables
    bool valid_a = false;
    PyObject* a_arg = NULL;
    char a_char = 0;
    PyArrayObject* a_arr = NULL;
    flint a[3];
    // angle variable
    bool valid_th = false;
    PyObject* th_arg = NULL;
    flint th;
    // center variable
    bool use_center = false;
    PyObject* c_arg = NULL;
    PyArrayObject* c_arr = NULL;
    flint c[3];
    NEW_AFFINE(out_arr, out_data);

    // allocate new affine transform objectc
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "OO|$O", rotation_keywords, &a_arg, &th_arg, &c_arg)) {
        Py_INCREF(a_arg);
        Py_INCREF(th_arg);
        // Get the axis argument
        if (PyUnicode_Check(a_arg)) {
            if (PyUnicode_GetLength(a_arg) == 1) {
                a_char = (char) *PyUnicode_1BYTE_DATA(a_arg);
                switch(a_char) {
                    case 'x':
                    case 'X': {
                        a_char = 'x';
                        valid_a = true;
                        break;
                    }
                    case 'y':
                    case 'Y': {
                        a_char = 'y';
                        valid_a = true;
                        break;
                    }
                    case 'z':
                    case 'Z': {
                        a_char = 'z';
                        valid_a = true;
                        break;
                    }
                }
            }
        } else {
            a_arr = (PyArrayObject*) PyArray_FROM_OT(a_arg, NPY_FLINT);
            if (a_arr != NULL) {
                if (PyArray_NDIM(a_arr) == 1) {
                    if (PyArray_SHAPE(a_arr)[0] == 3) {
                        for (i=0; i<3; i++) {
                            a[i] = *((flint*) PyArray_GETPTR1(a_arr, i));
                        }
                        valid_a = true;
                    }
                }
            }
        }
        // Get the angle argument
        if (PyLong_Check(th_arg)) {
            th = int_to_flint(PyLong_AsLongLong(th_arg));
            valid_th = true;
        }
        else if (PyFloat_Check(th_arg)) {
            th = double_to_flint(PyFloat_AsDouble(th_arg));
            valid_th = true;
        }
        else if (PyFlint_Check(th_arg)) {
            th = ((PyFlint*) th_arg)->obval;
            valid_th = true;
        }
        Py_DECREF(a_arg);
        Py_DECREF(th_arg);       
        if (c_arg != NULL) {
            Py_INCREF(c_arg);
            c_arr = (PyArrayObject*) PyArray_FROM_OT(c_arg, NPY_FLINT);
            if (c_arr != NULL) {
                if (PyArray_NDIM(c_arr) == 1) {
                    if (PyArray_SHAPE(c_arr)[0] == 3) {
                        for (i=0; i<3; i++) {
                            c[i] = *((flint*) PyArray_GETPTR1(c_arr, i));
                        }
                        use_center = true;
                    }
                }
                Py_DECREF(c_arr);
            }
            if (!use_center) {
                PyErr_SetString(PyExc_ValueError, "center must be a 3-length position [cx, cy, cz]");
                Py_DECREF(out_arr);
                Py_DECREF(c_arg);
                return NULL;
            }
            Py_DECREF(c_arg);
        }
    }
    if (!valid_a) {
        PyErr_SetString(PyExc_ValueError, "axis must be either a single character 'x','y','z' or 3-length axis [ax, ay, az]");
        Py_DECREF(out_arr);
        return NULL;
    }
    else if (!valid_th) {
        PyErr_SetString(PyExc_ValueError, "angle must be a numeric value");
        Py_DECREF(out_arr);
        return NULL;
    }
    switch(a_char) {
        case 0: {
            affine_set_rotaa(out_data, a, th);
            break;
        }
        case 'x': {
            affine_set_rotx(out_data, th);
            break;
        }
        case 'y': {
            affine_set_roty(out_data, th);
            break;
        }
        case 'z': {
            affine_set_rotz(out_data, th);
            break;
        }
    }
    if (use_center) {
        affine_relocate_center(out_data, c);
    }
    return (PyObject*) out_arr;
}

/// @brief Create a new pure reflection AffineTransform
PyDoc_STRVAR( reflection_docstring, "\
Create a new pure reflection transformation.\n\
\n\
:param normal: The character 'x','y','z' or a 3 length [ux, uy, uz] vector for\n\
    the normal vector for the reflection plane.\n\
:param center: Optional 3-length center position [cx, cy, cz] a point on the\n\
    plane of reflection operation.\n\
\n\
:return: A skew AffineTransformation.");
static PyObject* pyaffine_reflection(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* reflection_keywords[] = {"normal", "center", NULL};
    int i;
    // axis variables
    bool valid_n = false;
    PyObject* n_arg = NULL;
    char n_char = 0;
    PyArrayObject* n_arr = NULL;
    flint n[3];
    // center variable
    bool use_center = false;
    PyObject* c_arg = NULL;
    PyArrayObject* c_arr = NULL;
    flint c[3];
    NEW_AFFINE(out_arr, out_data);

    // allocate new affine transform objectc
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O", reflection_keywords, &n_arg, &c_arg)) {
        Py_INCREF(n_arg);
        // Get the normal vector argument
        if (PyUnicode_Check(n_arg)) {
            if (PyUnicode_GetLength(n_arg) == 1) {
                n_char = (char) *PyUnicode_1BYTE_DATA(n_arg);
                switch(n_char) {
                    case 'x':
                    case 'X': {
                        n_char = 'x';
                        valid_n = true;
                        break;
                    }
                    case 'y':
                    case 'Y': {
                        n_char = 'y';
                        valid_n = true;
                        break;
                    }
                    case 'z':
                    case 'Z': {
                        n_char = 'z';
                        valid_n = true;
                        break;
                    }
                }
            }
        } else {
            n_arr = (PyArrayObject*) PyArray_FROM_OT(n_arg, NPY_FLINT);
            if (n_arr != NULL) {
                if (PyArray_NDIM(n_arr) == 1) {
                    if (PyArray_SHAPE(n_arr)[0] == 3) {
                        for (i=0; i<3; i++) {
                            n[i] = *((flint*) PyArray_GETPTR1(n_arr, i));                            
                        }
                        valid_n = true;
                    }
                }
            }
        }
        Py_DECREF(n_arg);
        // Get the center argument
        if (c_arg != NULL) {
            Py_INCREF(c_arg);
            c_arr = (PyArrayObject*) PyArray_FROM_OT(c_arg, NPY_FLINT);
            if (c_arr != NULL) {
                if (PyArray_NDIM(c_arr) == 1) {
                    if (PyArray_SHAPE(c_arr)[0] == 3) {
                        for (i=0; i<3; i++) {
                            c[i] = *((flint*) PyArray_GETPTR1(c_arr, i));
                        }
                        use_center = true;
                    }
                }
                Py_DECREF(c_arr);
            }
            if (!use_center) {
                PyErr_SetString(PyExc_ValueError, "center must be a 3-length position [cx, cy, cz]");
                Py_DECREF(out_arr);
                Py_DECREF(c_arg);
                return NULL;
            }
            Py_DECREF(c_arg);
        }
    }
    if (!valid_n) {
        PyErr_SetString(PyExc_ValueError, "normal must be either a single character 'x','y','z' or 3-length axis [nx, ny, nz]");
        Py_DECREF(out_arr);
        return NULL;
    }
    
    switch (n_char) {
        case 0: {
            affine_set_refl_u(out_data, n);
            break;
        }
        case 'x': {
            affine_set_refl_yz(out_data);
            break;
        }
        case 'y': {
            affine_set_refl_zx(out_data);
            break;
        }
        case 'z': {
            affine_set_refl_xy(out_data);
            break;
        }
    }
    if (use_center) {
        affine_relocate_center(out_data, c);
    }
    return (PyObject*) out_arr;
}

/// @brief Create a new pure axis aligned skew AffineTransform
PyDoc_STRVAR( skew_docstring, "\
Create a new pure skew transformation.\n\
\n\
:param n: The character 'x','y','z' or a 3 length [nx, ny, nz] normal\n\
    vector to define the skew (shear) plane.\n\
:param s: A 3 length [sx, sy, sz] vector for the skew direction.\n\
:param center: Optional 3-length center position [cx, cy, cz] for the center of\n\
    the skew operation.\n\
\n\
:return: A skew AffineTransformation.");
static PyObject* pyaffine_skew(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* skew_keywords[] = {"n", "s", "center", NULL};
    int i;
    // skew plane normal variables
    bool valid_n = false;
    PyObject* n_arg = NULL;
    char n_char = 0;
    PyArrayObject* n_arr = NULL;
    flint n[3];
    // skew direction variables
    bool valid_s = false;
    PyObject* s_arg = NULL;
    PyArrayObject* s_arr = NULL;
    flint s[3];
    // center variable
    bool use_center = false;
    PyObject* c_arg = NULL;
    PyArrayObject* c_arr = NULL;
    flint c[3];
    // allocate new affine transform objectc
    NEW_AFFINE(out_arr, out_data);

    if (PyArg_ParseTupleAndKeywords(args, kwargs, "OO|$O", skew_keywords, &n_arg, &s_arg, &c_arg)) {
        Py_INCREF(n_arg);
        Py_INCREF(s_arg);
        // Get the skew plane normal argument
        if (PyUnicode_Check(n_arg)) {
            if (PyUnicode_GetLength(n_arg) == 1) {
                n_char = (char) *PyUnicode_1BYTE_DATA(n_arg);
                for (i=0; i<3; i++) {
                    n[i] = int_to_flint(0);
                }
                switch(n_char) {
                    case 'x':
                    case 'X': {
                        n[0] = int_to_flint(1);
                        valid_n = true;
                        break;
                    }
                    case 'y':
                    case 'Y': {
                        n[1] = int_to_flint(1);
                        valid_n = true;
                        break;
                    }
                    case 'z':
                    case 'Z': {
                        n[2] = int_to_flint(1);
                        valid_n = true;
                        break;
                    }
                }
            }
        } else {
            n_arr = (PyArrayObject*) PyArray_FROM_OT(n_arg, NPY_FLINT);
            if (n_arr != NULL) {
                if (PyArray_NDIM(n_arr) == 1) {
                    if (PyArray_SHAPE(n_arr)[0] == 3) {
                        for (i=0; i<3; i++) {
                            n[i] = *((flint*) PyArray_GETPTR1(n_arr, i));                            
                        }
                        valid_n = true;
                    }
                }
            }
        }
        // Get the skew direction argument
        s_arr = (PyArrayObject*) PyArray_FROM_OT(s_arg, NPY_FLINT);
        if (s_arr != NULL) {
            if (PyArray_NDIM(s_arr) == 1) {
                if (PyArray_SHAPE(s_arr)[0] == 3) {
                    for (i=0; i<3; i++) {
                        s[i] = *((flint*) PyArray_GETPTR1(s_arr, i));                            
                    }
                    valid_s = true;
                }
            }
        }
        Py_DECREF(s_arg);
        Py_DECREF(n_arg);

        // Get the center argument
        if (c_arg != NULL) {
            Py_INCREF(c_arg);
            c_arr = (PyArrayObject*) PyArray_FROM_OT(c_arg, NPY_FLINT);
            if (c_arr != NULL) {
                if (PyArray_NDIM(c_arr) == 1) {
                    if (PyArray_SHAPE(c_arr)[0] == 3) {
                        for (i=0; i<3; i++) {
                            c[i] = *((flint*) PyArray_GETPTR1(c_arr, i));
                        }
                        use_center = true;
                    }
                }
                Py_DECREF(c_arr);
            }
            if (!use_center) {
                PyErr_SetString(PyExc_ValueError, "center must be a 3-length position [cx, cy, cz]");
                Py_DECREF(out_arr);
                Py_DECREF(c_arg);
                return NULL;
            }
            Py_DECREF(c_arg);
        }
    }
    if (!valid_n) {
        PyErr_SetString(PyExc_ValueError, "skew normal must be either a single character 'x','y','z' or 3-length axis [nx, ny, nz]");
        Py_DECREF(out_arr);
        return NULL;
    }
    if (!valid_s) {
        PyErr_SetString(PyExc_ValueError, "skew direction must be a 3-length vector [sx, sy, sz]");
        Py_DECREF(out_arr);
        return NULL;
    }
    affine_set_skew(out_data, n, s);
    if (use_center) {
        affine_relocate_center(out_data, c);
    }
    return (PyObject*) out_arr;
}

// /// @brief Combine two affine transforms through matrix multiplication
// /// @param O_lhs The left hand side affine transform object
// /// @param O_rhs The right hand side affine transform object
// /// @return A new affine transform object combining the two inputs
// static PyObject* pyaffine_apply_transform(PyObject* O_lhs, PyObject* O_rhs) {
//     int i, j, k;
//     flint *res;
//     flint* lhs = ((PyAffine*) O_lhs)->array;
//     flint* rhs = ((PyAffine*) O_rhs)->array;
//     PyAffine* result = (PyAffine*) pyaffine_new(&PyAffine_Type, NULL, NULL);
//     if (result == NULL) {
//         PyErr_SetString(PyExc_SystemError, "Error allocating result AffineTransform");
//         return NULL;
//     }
//     result->type = AT_GENERIC;
//     res = result->array;
//     for (i=0; i<4; i++) {
//         for (j=0; j<4; j++) {
//             res[4*i+j] = int_to_flint(0);
//             for (k=0; k<4; k++) {
//                 flint_inplace_add(&(res[4*i+j]), flint_multiply(lhs[4*i+k], rhs[4*k+j]));
//             }
//         }
//     }
//     return (PyObject*) result;
// }
/// @brief Helper function for the affine apply macros
static inline flint flint_identity(flint f){ return f; }

// "(4,4),(3) -> (3)"
static void pyaffine_apply_vert(char** args, 
                                npy_intp const* dims,
                                npy_intp const* strides,
                                void* data) {
    npy_intp i, j, n;
    npy_intp N = dims[0];
    // npy_intp four = dims[1];
    // npy_intp three = dims[2];
    char* af_base = args[0];
    char* af_i;
    char* af;
    char* v_in_base = args[1];
    char* v_in;
    char* v_out_base = args[2];
    char* v_out;
    npy_intp d_af_n = strides[0];
    npy_intp d_v_in_n = strides[1];
    npy_intp d_v_out_n = strides[2];
    npy_intp d_af_i = strides[3];
    npy_intp d_af_j = strides[4];
    npy_intp d_v_in_j = strides[5];
    npy_intp d_v_out_i = strides[6];        
    flint v_in_f, w;
    for (n=0; n<N; n++) {
        // Matrix mult -> v_out = af(:3,:3).v_in
        for (i=0; i<3; i++) {
            af_i = af_base + i*d_af_i;
            v_out = v_out_base + i*d_v_out_i;
            *((flint*) v_out) = int_to_flint(0);
            for (j=0; j<3; j++) {
                af = af_i + j*d_af_j;
                v_in = v_in_base + j*d_v_in_j;
                v_in_f = flint_identity(*((flint*) v_in));
                flint_inplace_add((flint*) v_out, flint_multiply(*((flint*) af), v_in_f));
            }
            // Add trans -> v_out = v_out + af(:3,4)
            af = af_i + 3*d_af_j;
            flint_inplace_add((flint*) v_out, *((flint*) af));
        }
        // calc homogenous 'w' term
        af_i = af_base + 3*d_af_i;
        w = int_to_flint(0);
        for (j=0; j<3; j++) {
            af = af_i + j*d_af_j;
            v_in = v_in_base + j*d_v_in_j;
            v_in_f = flint_identity(*((flint*) v_in));
            flint_inplace_add(&w, flint_multiply(*((flint*) af), v_in_f));
        }
        af = af_i + 3*d_af_j;
        flint_inplace_add(&w, *((flint*) af));
        // rescale
        if (!flint_eq(w, int_to_flint(0))) {
            for (i=0; i<3; i++) {
                v_out = v_out_base + i*d_v_out_i;
                flint_inplace_divide((flint*) v_out, w);
            }
        }
        af_base += d_af_n;
        v_in_base += d_v_in_n;
        v_out_base += d_v_out_n;
    }
}

// pyaffine_rescale_homo
// "(4) -> (4)"
PyDoc_STRVAR(rescale_docstring, "\
Rescale an array of 4-length homogenous coordinates x,y,z,w -> x/w,y/w,z/w,1");
static void pyaffine_rescale_homo(char** args, 
                                  npy_intp const* dims,
                                  npy_intp const* strides,
                                  void* data) {
    npy_intp i, n;
    npy_intp N = dims[0];
    char* h_in_base = args[0];
    char* h_in;
    char* h_out_base = args[1];
    char* h_out;
    npy_intp d_h_in_n = strides[0];
    npy_intp d_h_out_n = strides[1];
    npy_intp d_h_in_i = strides[2];
    npy_intp d_h_out_i = strides[3];
    flint w;

    for (n=0; n<N; n++) {
        w = *((flint*) (h_in_base + 3*d_h_in_i));
        if (!flint_eq(w, int_to_flint(1))) {
            for( i=0; i<3; i++) {
                h_in = h_in_base + i*d_h_in_i;
                h_out = h_out_base + i*d_h_out_i;
                *((flint*) h_out) = flint_divide(*((flint*) h_in), w);
            }
            h_out = h_out_base + 3*d_h_out_i;
            *((flint*) h_out) = int_to_flint(1);
        }
        h_in_base += d_h_in_n;
        h_out_base += d_h_out_n;
    }
}

// /// @brief Apply the affine transformation to a 3xN? numpy array of flints
// /// This 
// static PyObject* pyaffine_apply_vertices(PyObject* trans, PyArrayObject* in_arr) {
//     /// Affine transform 4x4 matrix
//     flint* trans_data = ((PyAffine*) trans)->array;
//     /// Input array variables
//     int ndim = PyArray_NDIM(in_arr);
//     npy_intp* shape = PyArray_SHAPE(in_arr);
//     /// Outer loop variables
//     int i;
//     int num_verts = 1;
//     npy_intp outer_stride = PyArray_STRIDE(in_arr, 0)/sizeof(flint);
//     // Inner loop variables
//     int j, k;
//     npy_intp inner_stride = PyArray_STRIDE(in_arr, ndim-1)/sizeof(flint);
//     flint homo[4];
//     flint one = int_to_flint(1);
//     // working data arrays and new object
//     flint* in_data = (flint*) PyArray_DATA(in_arr);
//     flint* res_data;
//     PyArrayObject* result = (PyArrayObject*) PyArray_NewLikeArray(in_arr, NPY_KEEPORDER, NULL, 0);
//     if (result == NULL) {
//         PyErr_SetString(PyExc_SystemError, "Could not create output array");
//         return NULL;
//     }
//     res_data = (flint*) PyArray_DATA(result);
//     // Get the number of vertices we're operating on
//     for (i=1; i<ndim; i++) {
//         num_verts *= shape[i];
//     }
//     // Loop over the vertices
//     for (i=1; i<num_verts; i++) {
//         // Do the matrix-vector multiplication
//         for (j=0; j<4; j++) {
//             homo[j] = int_to_flint(0);
//             for (k=0; k<3; k++) {
//                 flint_inplace_add(&(homo[j]), flint_multiply(trans_data[4*j+k], in_data[k*outer_stride+i*inner_stride]));
//             }
//             flint_inplace_add(&(homo[j]), trans_data[4*j+3]);
//         }
//         // Possibly rescale homogenous coordinates
//         if (!flint_eq(homo[3], one)) {
//             for (j=0; j<3; j++) {
//                 flint_inplace_divide(&(homo[j]), homo[3]);
//             }
//         }
//         // Copy data to output
//         for (j=0; j<3; j++) {
//             res_data[j*outer_stride+i*inner_stride] = homo[j];
//         }        
//     }

//     return (PyObject*) result;
// }

// /// @brief Apply the affine transformation to a 4xN? numpy array of flints
// static PyObject* pyaffine_apply_homogenous(PyObject* trans, PyArrayObject* in_arr) {
//     /// Affine transform 4x4 matrix
//     flint* trans_data = ((PyAffine*) trans)->array;
//     /// Input array variables
//     int ndim = PyArray_NDIM(in_arr);
//     npy_intp* shape = PyArray_SHAPE(in_arr);
//     /// Outer loop variables
//     int i;
//     int num_verts = 1;
//     npy_intp outer_stride = PyArray_STRIDE(in_arr, 0)/sizeof(flint);
//     // Inner loop variables
//     int j, k;
//     npy_intp inner_stride = PyArray_STRIDE(in_arr, ndim-1)/sizeof(flint);
//     flint one = int_to_flint(1);
//     // working data arrays and new object
//     flint* homo;
//     flint* in_data = (flint*) PyArray_DATA(in_arr);
//     flint* res_data;
//     PyArrayObject* result = (PyArrayObject*) PyArray_NewLikeArray(in_arr, NPY_KEEPORDER, NULL, 0);
//     if (result == NULL) {
//         PyErr_SetString(PyExc_SystemError, "Could not create output array");
//         return NULL;
//     }
//     res_data = (flint*) PyArray_DATA(result);
//     // Get the number of vertices we're operating on
//     for (i=1; i<ndim; i++) {
//         num_verts *= shape[i];
//     }
//     // Loop over the vertices
//     for (i=1; i<num_verts; i++) {
//         // Do the matrix-vector multiplication
//         for (j=0; j<4; j++) {
//             homo[j] = int_to_flint(0);
//             for (k=0; k<4; k++) {
//                 flint_inplace_add(&(homo[j]), flint_multiply(trans_data[4*j+k], in_data[k*outer_stride+i*inner_stride]));
//             }
//         }
//         // Possibly rescale homogenous coordinates
//         if (!flint_eq(homo[3], one)) {
//             for (j=0; j<3; j++) {
//                 flint_inplace_divide(&(homo[j]), homo[3]);
//             }
//             homo[3] = one;
//         }
//         // Copy data to output
//         for (j=0; j<4; j++) {
//             res_data[j*outer_stride+i*inner_stride] = homo[j];
//         }        
//     }

//     return (PyObject*) result;
// }

/// @brief Apply the affine transformation to an applicable object
// static PyObject* pyaffine_apply(PyObject* self, PyObject* arg) {
//     PyAffine* at_in = NULL;
//     PyAffine* at_out = NULL;;
//     PyArrayObject* arr_in = NULL;    
//     PyArrayObject* arr_out = NULL;
//     int nd = 0;
//     npy_intp* arr_out_shape = NULL;
//     PyAffine* at_self = (PyAffine*) self;
//     int ret = -1;
//     int type_num = 0;
//     bool new_array = false;
//     // Use transform combination if argument is something
//     if (PyAffine_Check(arg)) {
//         at_in = (PyAffine*) arg;
//         Py_INCREF(at_in);
//         at_out = (PyAffine*) pyaffine_new(&PyAffine_Type, NULL, NULL);
//         if (at_out == NULL) {
//             PyErr_SetString(PyExc_SystemError, "Error allocating new AffineTransform");
//             Py_DECREF(at_in);
//             return NULL;
//         }
//         affine_combine(at_out->array, at_self->array, at_in->array);
//         Py_DECREF(at_in);
//         return (PyObject*) at_out;
//     }
//     else if (PyObject_IsInstance(arg, (PyObject*) &PyArray_Type)) {
//         arr_in = (PyArrayObject*) arg;
//         Py_INCREF(arr_in);
//         nd = PyArray_NDIM(arr_in);
//         if (nd < 1) {
//             PyErr_SetString(PyExc_ValueError, "Numpy argument cannot be an array scalar");
//             Py_DECREF(arr_in);
//             return NULL;
//         }
//         arr_out_shape = PyArray_SHAPE(arr_in);
//         if (arr_out_shape[0] != 3 || arr_out_shape[0] != 4) {
//             PyErr_SetString(PyExc_ValueError, "Numpy array argument should be a 3x? or 4x? array");
//             Py_DECREF(arr_in);
//             return NULL;
//         }
//         arr_out = (PyArrayObject*) PyArray_SimpleNew(nd, arr_out_shape, NPY_FLINT);
//         if (arr_out == NULL) {
//             PyErr_SetString(PyExc_SystemError, "Error allocating output array");
//             Py_DECREF(arr_in);
//             return NULL;
//         }
//         type_num = PyArray_TYPE(arr_in);
//         if ( type_num == NPY_INT32 ) {
//             if (arr_out_shape[0] == 3) {
//                 ret = affine_apply_vert_int(
//                     (flint*) PyArray_DATA(arr_out),
//                     at_self->array,
//                     (int*) PyArray_DATA(arr_in),
//                     PyArray_NDIM(arr_in),
//                     PyArray_SHAPE(arr_in),
//                     PyArray_STRIDES(arr_in)
//                 );
//             } else {
//                 ret = affine_apply_homo_int(
//                     (flint*) PyArray_DATA(arr_out),
//                     at_self->array,
//                     (int*) PyArray_DATA(arr_in),
//                     PyArray_NDIM(arr_in),
//                     PyArray_SHAPE(arr_in),
//                     PyArray_STRIDES(arr_in)
//                 );
//             }
//         }
//         else if ( type_num == NPY_FLOAT64 ) {
//             if (arr_out_shape[0] == 3) {
//                 ret = affine_apply_vert_double(
//                     (flint*) PyArray_DATA(arr_out),
//                     at_self->array,
//                     (double*) PyArray_DATA(arr_in),
//                     PyArray_NDIM(arr_in),
//                     PyArray_SHAPE(arr_in),
//                     PyArray_STRIDES(arr_in)
//                 );
//             } else {
//                 ret = affine_apply_homo_double(
//                     (flint*) PyArray_DATA(arr_out),
//                     at_self->array,
//                     (double*) PyArray_DATA(arr_in),
//                     PyArray_NDIM(arr_in),
//                     PyArray_SHAPE(arr_in),
//                     PyArray_STRIDES(arr_in)
//                 );
//             }
//         }
//         else if ( type_num == NPY_FLINT ) {
//             if (arr_out_shape[0] == 3) {
//                 ret = affine_apply_vert_flint(
//                     (flint*) PyArray_DATA(arr_out),
//                     at_self->array,
//                     (flint*) PyArray_DATA(arr_in),
//                     PyArray_NDIM(arr_in),
//                     PyArray_SHAPE(arr_in),
//                     PyArray_STRIDES(arr_in)
//                 );
//             } else {
//                 ret = affine_apply_homo_flint(
//                     (flint*) PyArray_DATA(arr_out),
//                     at_self->array,
//                     (flint*) PyArray_DATA(arr_in),
//                     PyArray_NDIM(arr_in),
//                     PyArray_SHAPE(arr_in),
//                     PyArray_STRIDES(arr_in)
//                 );
//             }
//         }
//         else {
//             arr_in = (PyArrayObject*) PyArray_FROM_OT(arg, NPY_FLINT);
//             if (arr_in == NULL) {
//                 PyErr_SetString(PyExc_TypeError, "Could not read input as 3x? or 4x? array of numbers");
//                 Py_DECREF(arg);
//                 Py_DECREF(arr_out);
//                 return NULL;
//             }
//             if (arr_out_shape[0] == 3) {
//                 ret = affine_apply_vert_flint(
//                     (flint*) PyArray_DATA(arr_out),
//                     at_self->array,
//                     (flint*) PyArray_DATA(arr_in),
//                     PyArray_NDIM(arr_in),
//                     PyArray_SHAPE(arr_in),
//                     PyArray_STRIDES(arr_in)
//                 );
//             } else {
//                 ret = affine_apply_homo_flint(
//                     (flint*) PyArray_DATA(arr_out),
//                     at_self->array,
//                     (flint*) PyArray_DATA(arr_in),
//                     PyArray_NDIM(arr_in),
//                     PyArray_SHAPE(arr_in),
//                     PyArray_STRIDES(arr_in)
//                 );
//             }
//         }
//         if (ret < 0) {
//             PyErr_SetString(PyExc_ValueError, "Array has too many (>10) dimensions");
//             Py_DECREF(arr_in);
//             Py_DECREF(arr_out);
//             return NULL;
//         }
//         Py_DECREF(arr_in);
//     } else {
//         arr_in = (PyArrayObject*) PyArray_FROM_OT(arg, NPY_FLINT);
//         if (arr_in == NULL) {
//             PyErr_SetString(PyExc_TypeError, "Could not read input as 3x? or 4x? array of numbers");
//             Py_DECREF(arg);
//             Py_DECREF(arr_out);
//             return NULL;
//         }
//         Py_INCREF(arr_in);
//         nd = PyArray_NDIM(arr_in);
//         if (nd < 1) {
//             PyErr_SetString(PyExc_ValueError, "Argument should be an affine transform or 3x? or 4x? array");
//             Py_DECREF(arr_in);
//             return NULL;
//         }
//         arr_out_shape = PyArray_SHAPE(arr_in);
//         if (arr_out_shape[0] != 3 || arr_out_shape[0] != 4) {
//             PyErr_SetString(PyExc_ValueError, "Argument should be an affine transform or 3x? or 4x? array");
//             Py_DECREF(arr_in);
//             return NULL;
//         }
//         arr_out = (PyArrayObject*) PyArray_SimpleNew(nd, arr_out_shape, NPY_FLINT);
//         if (arr_out == NULL) {
//             PyErr_SetString(PyExc_SystemError, "Error allocating output array");
//             Py_DECREF(arr_in);
//             return NULL;
//         }
//         if (arr_out_shape[0] == 3) {
//             ret = affine_apply_vert_flint(
//                 (flint*) PyArray_DATA(arr_out),
//                 at_self->array,
//                 (flint*) PyArray_DATA(arr_in),
//                 PyArray_NDIM(arr_in),
//                 PyArray_SHAPE(arr_in),
//                 PyArray_STRIDES(arr_in)
//             );
//         } else {
//             ret = affine_apply_homo_flint(
//                 (flint*) PyArray_DATA(arr_out),
//                 at_self->array,
//                 (flint*) PyArray_DATA(arr_in),
//                 PyArray_NDIM(arr_in),
//                 PyArray_SHAPE(arr_in),
//                 PyArray_STRIDES(arr_in)
//             );
//         }
//         if (ret < 0) {
//             PyErr_SetString(PyExc_ValueError, "Array has too many (>10) dimensions");
//             Py_DECREF(arr_in);
//             Py_DECREF(arr_out);
//             return NULL;
//         }
//         Py_DECREF(arr_in);
//     }
//     return (PyObject*) arr_out;
// }

static PyMethodDef AffineMethods[] = {
    {"eye", pyaffine_eye, METH_NOARGS, eye_docstring},
    {"from_mat", pyaffine_from_mat, METH_O, from_mat_docstring},
    {"trans", (PyCFunction) pyaffine_translation, 
    METH_VARARGS | METH_KEYWORDS, translation_docstring},
    {"scale", (PyCFunction) pyaffine_scale,
    METH_VARARGS | METH_KEYWORDS, scale_docstring},
    {"rot", (PyCFunction) pyaffine_rotation,
    METH_VARARGS | METH_KEYWORDS, rotation_docstring},
    {"refl", (PyCFunction) pyaffine_reflection,
    METH_VARARGS | METH_KEYWORDS, reflection_docstring},
    {"skew", (PyCFunction) pyaffine_skew,
    METH_VARARGS | METH_KEYWORDS, skew_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_c_affine",
    .m_doc = "Affine Transforms",
    .m_size = -1,
    .m_methods = AffineMethods,
};

/// @brief The module initialization function
PyMODINIT_FUNC PyInit__c_affine(void) {
    PyObject* m;
    PyObject* rescale_ufunc;
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "Could not create affine module.");
        return NULL;
    }
    // Import and initialize numpy
    import_array();
    if (PyErr_Occurred()) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "Could not initialize NumPy.");
        return NULL;
    }
    // Import flint c API
    if (import_flint() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "Count not load flint c API");
        return NULL;
    }
    // Import numpys ufunc api
    import_ufunc();
    if (PyErr_Occurred()) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "Could not load NumPy ufunc c API.");
        return NULL;
    }
    PyUFuncGenericFunction pyaffine_rescale_funcs[] = {
        pyaffine_rescale_homo
    };
    void* pyaffine_rescale_data[] = {NULL};
    char pyaffine_rescale_types[] = {NPY_FLINT, NPY_FLINT};
    char pyaffine_rescale_sig[] = "(4)->(4)";
    rescale_ufunc = PyUFunc_FromFuncAndDataAndSignature(
        pyaffine_rescale_funcs,
        pyaffine_rescale_data,
        pyaffine_rescale_types,
        1,
        1,
        1,
        PyUFunc_None,
        "rescale",
        rescale_docstring,
        NULL,
        pyaffine_rescale_sig
    );

    return m;
}