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

#include <flint.h>
#include <numpy_flint.h>

#define AFFINE_MODULE
#include "affine.h"
#include "pyaffine.h"

/// @brief The __new__ allocating constructor
/// @param type The type of the PyObject
/// @return A new PyObject of type `type`
static PyObject* pyaffine_new(PyTypeObject* type, 
                              PyObject* NPY_UNUSED(args),
                              PyObject* NPY_UNUSED(kwargs)) {
    PyAffine* self = (PyAffine*) type->tp_alloc(type, 0);
    return (PyObject*) self;
}

/// @brief The __init__ initializing constructor
/// @param self The object to be initialized
/// @param args Unused positional argument tuple
/// @param kwargs Unused keyword argument dict
/// @return 0 on success, -1 on failure
static int pyaffine_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyAffine* at_self = (PyAffine*) self;

    if ((PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs))){
        PyErr_SetString(PyExc_TypeError,
                        "AffineTrans constructor doesn't take any arguments");
        return -1;
    }
    affine_eye(at_self->array);
    return 0;
}

/// @brief The __str__ printing method
/// @return A python string representation of the tracked value
static PyObject* pyaffine_str(PyObject* self) {
    PyAffine* at_self = (PyAffine*) self;
    switch(at_self->type) {
        case AT_TRANSLATION:
            return PyUnicode_FromString("Translation");
            break;
        case AT_ROTATION:
            return PyUnicode_FromString("Rotation");
            break;
        case AT_SCALE:
            return PyUnicode_FromString("Scale");
            break;
        case AT_REFLECTION:
            return PyUnicode_FromString("Reflection");
            break;
        case AT_SKEW:
            return PyUnicode_FromString("SkewTransform");
            break;
        default:
            break;
    }
    return PyUnicode_FromString("AffineTransform");
}

/// @brief The __hash__ function create an unique-ish integer from the flint.
///        Implements Bob Jenkin's one-at-a-time hash.
/// @return An integer to be used in a hash-table
static Py_hash_t pyaffine_hash(PyObject *self) {
    PyAffine* at_self = (PyAffine*) self;
    uint8_t* array_as_bytes = (uint8_t*) at_self->array;
    size_t i = 0;
    Py_hash_t h = 0;
    for (i=0; i<(16*sizeof(flint)); ++i) {
        h += array_as_bytes[i];
        h += h << 10;
        h ^= h >> 6;
    }
    h += h << 3;
    h ^= h >> 11;
    h += h << 15;
    return (h==-1)?2:h;
}

/// @brief Get the size of the interval of flint object
/// This defines a member property getter for the size of the interval
/// so you can get the endpoints of hte interval with `eps = f.eps`
static const char get_array_docstring[] = "\
The 4x4 matrix representation of the transformation.\n\
\n\
:getter: Returns a 4x4 numpy array of flints\n\
:setter: Not implemented";
static PyObject* pyaffine_get_array(PyObject *self, void *NPY_UNUSED(closure)) {
    PyAffine* at_self = (PyAffine*) self;
    // Create a new 4x4 numpy array
    int nd = 2;
    npy_intp dims[2] = {4,4};
    PyObject* arr = PyArray_SimpleNewFromData(nd, dims, NPY_FLINT, (void*) at_self->array);
    if (arr == NULL) {
        PyErr_SetString(PyExc_SystemError, "Could not get create new numpy array of flints");
        return NULL;
    }
    return arr;
}

/// @brief Defines the property getter/setter methods
static PyGetSetDef pyaffine_getset[] = {
    {"array", pyaffine_get_array, NULL,
    get_array_docstring, NULL},
    //sentinal
    {NULL, NULL, NULL, NULL, NULL}
};

/// @brief Set the array members from a 4x4 array
static void pyaffine_from_4x4(PyAffine* self, PyArrayObject* arr) {
    int i, j;
    for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
            self->array[4*i+j] = *((flint*) PyArray_GETPTR2(arr, i, j));
        }
    }
}

/// @brief Set the array members from a 4x3 array
static void pyaffine_from_3x4(PyAffine* self, PyArrayObject* arr) {
    int i, j;
    for (i=0; i<3; i++) {
        for (j=0; j<4; j++) {
            self->array[4*i+j] = *((flint*) PyArray_GETPTR2(arr, i, j));
        }
    }
    for(j=0; j<3; j++) {
        self->array[12+j] = int_to_flint(0);
    }
    self->array[15] = int_to_flint(1);
}

/// @brief Set the array members from a 3x3 array
static void pyaffine_from_3x3(PyAffine* self, PyArrayObject* arr) {
    int i, j;
    for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
            self->array[4*i+j] = *((flint*) PyArray_GETPTR2(arr, i, j));
        }
        self->array[4*i+3] = int_to_flint(0);
    }
    for(j=0; j<3; j++) {
        self->array[12+j] = int_to_flint(0);
    }
    self->array[15] = int_to_flint(1);
}

/// @brief Classmethod for making an AffineTransform from a matrix
static const char from_mat_docstring[] ="\
Create a new generic affine transform from a 4x4, 3x4 or 3x3 matrix\n\
\n\
* A 3x3 matrix will only specify the linear transformation.\n\
* A 3x4 matrix will specify the linear transformation and translation.\n\
* A 4x4 will specify the linear transformation, translation, and perspective\n\
    transformation.\n\
\n\
:param mat: The input matrix (any properly shaped nested sequence type).\n\
\n\
:return: An AffineTransform object corresponding to the matrix";
static PyObject* pyaffine_from_mat(PyObject* cls, PyObject* args) {
    PyAffine* at_ref = NULL;
    PyArrayObject* arr = NULL;
    PyObject* O = NULL;
    npy_intp* shape = NULL;
    if (PyArg_ParseTuple(args, "O", &O)) {
        Py_XINCREF(O);
        at_ref = (PyAffine*) pyaffine_new((PyTypeObject*) cls, NULL, NULL);
        if (at_ref == NULL) {
            PyErr_SetString(PyExc_SystemError, "Error allocating new AffineTransform");
            return NULL;
        }
        arr = (PyArrayObject*) PyArray_FROM_OT(O, NPY_FLINT);
        if (arr != NULL) {
            if (PyArray_NDIM(arr) == 2) {
                shape = PyArray_SHAPE(arr);
                if (shape[0] == 4 && shape[1] == 4) {
                    at_ref->type = AT_GENERIC;
                    pyaffine_from_4x4(at_ref, arr);
                    Py_DECREF(O);
                    Py_DECREF(arr);
                    return (PyObject*) at_ref;
                }
                if (shape[0] == 3 && shape[1] == 4) {
                    at_ref->type = AT_GENERIC;
                    pyaffine_from_3x4(at_ref, arr);
                    Py_DECREF(O);
                    Py_DECREF(arr);
                    return (PyObject*) at_ref;
                }
                if (shape[0] == 3 && shape[1] == 3) {
                    at_ref->type = AT_GENERIC;
                    pyaffine_from_3x3(at_ref, arr);
                    Py_DECREF(O);
                    Py_DECREF(arr);
                    return (PyObject*) at_ref;
                }
            }
        }
    }
    PyErr_SetString(PyExc_ValueError, "Argument must be a 4x4, 3x4 or 4x3 array-like object");
    Py_XDECREF(O);
    Py_XDECREF(arr);
    return NULL;
}

/// @brief Create a new pure translation AffineTransform
/// @param args The [x,y,z] coordinates of the translation
static const char translation_docstring[] = "\
Create a new pure translation transformation.\n\
\n\
:param d: A 3-length sequence [dx, dy, dz]\n\
:param center: Ignored\n\
\n\
:return: An pure translation AffineTransformation.";
static PyObject* pyaffine_translation(PyObject* cls, PyObject* args, PyObject* kwargs) {
    static char* translation_keywords[] = {"d", "center", NULL};
    int i;
    PyAffine* at_ref = NULL;
    PyObject* d_obj = NULL;
    PyArrayObject* d_arr = NULL;
    flint d[3];
    bool valid_d = false;
    PyObject* c_obj = NULL;

    at_ref = (PyAffine*) pyaffine_new((PyTypeObject*) cls, NULL, NULL);
    if (at_ref == NULL) {
        PyErr_SetString(PyExc_SystemError, "Error allocating new AffineTransform");
        return NULL;
    }
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O", translation_keywords, &d_obj, &c_obj)) {
        Py_INCREF(d_obj);
        d_arr = (PyArrayObject*) PyArray_FROM_OT(d_obj, NPY_FLINT);
        if (d_arr != NULL) {
            if (PyArray_NDIM(d_arr) == 1) {
                if (PyArray_SHAPE(d_arr)[0] == 3) {
                    for (i=0; i<3; i++) {
                        d[i] = *((flint*) PyArray_GETPTR1(d_arr, i));
                    }
                    valid_d = true;
                }
            }
            Py_DECREF(d_arr);
        }
        Py_DECREF(d_obj);
    }
    if (!valid_d) {
        PyErr_SetString(PyExc_ValueError, "Translation argument should be a 3 length sequence");
        Py_XDECREF(at_ref);
        return NULL;
    }
    at_ref->type = AT_TRANSLATION;
    affine_set_translation(at_ref->array, d);
    return (PyObject*) at_ref;
}

/// @brief Create a new pure scaling AffineTransform
static const char scale_docstring[] = "\
Create a new pure scaling transformation.\n\
\n\
:param s: A scalar or 3-length sequence [sx, sy, sz]\n\
:param center: Optional 3-length center position [cx, cy, cz] for the scaling\n\
    transform\n\
\n\
:return: A scaling if AffineTransformation.";
static PyObject* pyaffine_scale(PyObject* cls, PyObject* args, PyObject* kwargs) {
    static char* scale_keywords[] = {"s", "center", NULL};
    int i;
    long long n;
    double d;
    PyAffine* at_ref = NULL;
    // variable for scale
    bool valid_scale = false;
    PyObject* s_arg = NULL;
    PyArrayObject* s_arr = NULL;
    flint s[3] = {0};
    // variable for center
    bool use_center = false;
    PyObject* c_arg = NULL;
    PyArrayObject* c_arr = NULL;
    flint c[3] = {0};
    // allocate new affine transform PyObject
    at_ref = (PyAffine*) pyaffine_new((PyTypeObject*) cls, NULL, NULL);
    if (at_ref == NULL) {
        PyErr_SetString(PyExc_SystemError, "Error allocating new AffineTransform");
        return NULL;
    }
    // Parse args
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O", scale_keywords, &s_arg, &c_arg)) {
        // Temporarily keeping around to remind myself how to 'print debug'
        // printf("%s\n", Py_TYPE(s_arg)->tp_name);
        // printf("%d\n", PyFlint_Check(s_arg));
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
                s[i] = ((PyFlint*) s_arg)->obval;
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
                Py_DECREF(at_ref);
                Py_DECREF(c_arg);
                return NULL;
            }
            Py_DECREF(c_arg);
        }
    }
    if (!valid_scale) {
        PyErr_SetString(PyExc_ValueError, "s must be a scalar or scalar or 3-length non-uniform scaling [sx, sy, sz]");
        Py_DECREF(at_ref);
        return NULL;
    }
    at_ref->type = AT_SCALE;
    affine_set_scale(at_ref->array, s);
    if (use_center) {
        affine_relocate_center(at_ref->array, c);
    }
    return (PyObject*) at_ref;
}

/// @brief Create a new pure rotation AffineTransform
static const char rotation_docstring[] = "\
Create a new pure rotation transformation.\n\
\n\
:param axis: The character 'x','y','z' or a three length vector [ax, ay, az]\n\
:param angle: The angle in radians to rotate\n\
:param center: Optional 3-length position [cx, cy, cz] for to specify a point\n\
    on the axix of rotation\n\
\n\
:return: A rotation AffineTransformation.";
static PyObject* pyaffine_rotation(PyObject* cls, PyObject* args, PyObject* kwargs) {
    static char* rotation_keywords[] = {"axis", "angle", "center", NULL};
    int i;
    PyAffine* at_ref = NULL;
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

    // allocate new affine transform objectc
    at_ref = (PyAffine*) pyaffine_new((PyTypeObject*) cls, NULL, NULL);
    if (at_ref == NULL) {
        PyErr_SetString(PyExc_SystemError, "Error allocating new AffineTransform");
        return NULL;
    }
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
                Py_DECREF(at_ref);
                Py_DECREF(c_arg);
                return NULL;
            }
            Py_DECREF(c_arg);
        }
    }
    if (!valid_a) {
        PyErr_SetString(PyExc_ValueError, "axis must be either a single character 'x','y','z' or 3-length axis [ax, ay, az]");
        Py_DECREF(at_ref);
        return NULL;
    }
    else if (!valid_th) {
        PyErr_SetString(PyExc_ValueError, "angle must be a numeric value");
        Py_DECREF(at_ref);
        return NULL;
    }
    at_ref->type = AT_ROTATION;
    switch(a_char) {
        case 0: {
            affine_set_rotaa(at_ref->array, a, th);
            break;
        }
        case 'x': {
            affine_set_rotx(at_ref->array, th);
            break;
        }
        case 'y': {
            affine_set_roty(at_ref->array, th);
            break;
        }
        case 'z': {
            affine_set_rotz(at_ref->array, th);
            break;
        }
    }
    if (use_center) {
        affine_relocate_center(at_ref->array, c);
    }
    return (PyObject*) at_ref;
}

/// @brief Create a new pure reflection AffineTransform
static const char reflection_docstring[] = "\
Create a new pure reflection transformation.\n\
\n\
:param normal: The character 'x','y','z' or a 3 length [ux, uy, uz] vector for\n\
    the normal vector for the reflection plane.\n\
:param center: Optional 3-length center position [cx, cy, cz] a point on the\n\
    plane of reflection operation.\n\
\n\
:return: A skew AffineTransformation.";
static PyObject* pyaffine_reflection(PyObject* cls, PyObject* args, PyObject* kwargs) {
    static char* reflection_keywords[] = {"normal", "center", NULL};
    int i;
    PyAffine* at_ref = NULL;
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

    // allocate new affine transform objectc
    at_ref = (PyAffine*) pyaffine_new((PyTypeObject*) cls, NULL, NULL);
    if (at_ref == NULL) {
        PyErr_SetString(PyExc_SystemError, "Error allocating new AffineTransform");
        return NULL;
    }
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
                Py_DECREF(at_ref);
                Py_DECREF(c_arg);
                return NULL;
            }
            Py_DECREF(c_arg);
        }
    }
    if (!valid_n) {
        PyErr_SetString(PyExc_ValueError, "normal must be either a single character 'x','y','z' or 3-length axis [nx, ny, nz]");
        Py_DECREF(at_ref);
        return NULL;
    }
    
    switch (n_char) {
        case 0: {
            affine_set_refl_u(at_ref->array, n);
            break;
        }
        case 'x': {
            affine_set_refl_yz(at_ref->array);
            break;
        }
        case 'y': {
            affine_set_refl_zx(at_ref->array);
            break;
        }
        case 'z': {
            affine_set_refl_xy(at_ref->array);
            break;
        }
    }
    if (use_center) {
        affine_relocate_center(at_ref->array, c);
    }
    return (PyObject*) at_ref;
}

/// @brief Create a new pure axis aligned skew AffineTransform
static const char skew_docstring[] = "\
Create a new pure skew transformation.\n\
\n\
:param n: The character 'x','y','z' or a 3 length [nx, ny, nz] normal\n\
    vector to define the skew (shear) plane.\n\
:param s: A 3 length [sx, sy, sz] vector for the skew direction.\n\
:param center: Optional 3-length center position [cx, cy, cz] for the center of\n\
    the skew operation.\n\
\n\
:return: A skew AffineTransformation.";
static PyObject* pyaffine_skew(PyObject* cls, PyObject* args, PyObject* kwargs) {
    static char* skew_keywords[] = {"n", "s", "center", NULL};
    int i;
    PyAffine* at_ref = NULL;
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
    at_ref = (PyAffine*) pyaffine_new((PyTypeObject*) cls, NULL, NULL);
    if (at_ref == NULL) {
        PyErr_SetString(PyExc_SystemError, "Error allocating new AffineTransform");
        return NULL;
    }
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
                Py_DECREF(at_ref);
                Py_DECREF(c_arg);
                return NULL;
            }
            Py_DECREF(c_arg);
        }
    }
    if (!valid_n) {
        PyErr_SetString(PyExc_ValueError, "skew normal must be either a single character 'x','y','z' or 3-length axis [nx, ny, nz]");
        Py_DECREF(at_ref);
        return NULL;
    }
    if (!valid_s) {
        PyErr_SetString(PyExc_ValueError, "skew direction must be a 3-length vector [sx, sy, sz]");
        Py_DECREF(at_ref);
        return NULL;
    }
    at_ref->type = AT_SKEW;
    affine_set_skew(at_ref->array, n, s);
    if (use_center) {
        affine_relocate_center(at_ref->array, c);
    }
    return (PyObject*) at_ref;
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
static const char apply_docstring[] = "\
Apply the affine transform to the argument\n\
\n\
:param arg: An affine transform\n\
:return: An afffine transform with with combined\n\
\n\
:param arg: A 3x? array of vertex coordinates\n\
:return: A new array of transformed coordinates\n\
\n\
:param arg: A 4x? array of homogenous coordinates\n\
:return: A new array of transformed homogeneous coordinates";
static PyObject* pyaffine_apply(PyObject* self, PyObject* arg) {
    PyAffine* at_in = NULL;
    PyAffine* at_out = NULL;;
    PyArrayObject* arr_in = NULL;    
    PyArrayObject* arr_out = NULL;
    int nd = 0;
    npy_intp* arr_out_shape = NULL;
    PyAffine* at_self = (PyAffine*) self;
    int ret = -1;
    int type_num = 0;
    bool new_array = false;
    // Use transform combination if argument is something
    if (PyAffine_Check(arg)) {
        at_in = (PyAffine*) arg;
        Py_INCREF(at_in);
        at_out = (PyAffine*) pyaffine_new(&PyAffine_Type, NULL, NULL);
        if (at_out == NULL) {
            PyErr_SetString(PyExc_SystemError, "Error allocating new AffineTransform");
            Py_DECREF(at_in);
            return NULL;
        }
        affine_combine(at_out->array, at_self->array, at_in->array);
        Py_DECREF(at_in);
        return (PyObject*) at_out;
    }
    else if (PyObject_IsInstance(arg, (PyObject*) &PyArray_Type)) {
        arr_in = (PyArrayObject*) arg;
        Py_INCREF(arr_in);
        nd = PyArray_NDIM(arr_in);
        if (nd < 1) {
            PyErr_SetString(PyExc_ValueError, "Numpy argument cannot be an array scalar");
            Py_DECREF(arr_in);
            return NULL;
        }
        arr_out_shape = PyArray_SHAPE(arr_in);
        if (arr_out_shape[0] != 3 || arr_out_shape[0] != 4) {
            PyErr_SetString(PyExc_ValueError, "Numpy array argument should be a 3x? or 4x? array");
            Py_DECREF(arr_in);
            return NULL;
        }
        arr_out = (PyArrayObject*) PyArray_SimpleNew(nd, arr_out_shape, NPY_FLINT);
        if (arr_out == NULL) {
            PyErr_SetString(PyExc_SystemError, "Error allocating output array");
            Py_DECREF(arr_in);
            return NULL;
        }
        type_num = PyArray_TYPE(arr_in);
        if ( type_num == NPY_INT32 ) {
            if (arr_out_shape[0] == 3) {
                ret = affine_apply_vert_int(
                    (flint*) PyArray_DATA(arr_out),
                    at_self->array,
                    (int*) PyArray_DATA(arr_in),
                    PyArray_NDIM(arr_in),
                    PyArray_SHAPE(arr_in),
                    PyArray_STRIDES(arr_in)
                );
            } else {
                ret = affine_apply_homo_int(
                    (flint*) PyArray_DATA(arr_out),
                    at_self->array,
                    (int*) PyArray_DATA(arr_in),
                    PyArray_NDIM(arr_in),
                    PyArray_SHAPE(arr_in),
                    PyArray_STRIDES(arr_in)
                );
            }
        }
        else if ( type_num == NPY_FLOAT64 ) {
            if (arr_out_shape[0] == 3) {
                ret = affine_apply_vert_double(
                    (flint*) PyArray_DATA(arr_out),
                    at_self->array,
                    (double*) PyArray_DATA(arr_in),
                    PyArray_NDIM(arr_in),
                    PyArray_SHAPE(arr_in),
                    PyArray_STRIDES(arr_in)
                );
            } else {
                ret = affine_apply_homo_double(
                    (flint*) PyArray_DATA(arr_out),
                    at_self->array,
                    (double*) PyArray_DATA(arr_in),
                    PyArray_NDIM(arr_in),
                    PyArray_SHAPE(arr_in),
                    PyArray_STRIDES(arr_in)
                );
            }
        }
        else if ( type_num == NPY_FLINT ) {
            if (arr_out_shape[0] == 3) {
                ret = affine_apply_vert_flint(
                    (flint*) PyArray_DATA(arr_out),
                    at_self->array,
                    (flint*) PyArray_DATA(arr_in),
                    PyArray_NDIM(arr_in),
                    PyArray_SHAPE(arr_in),
                    PyArray_STRIDES(arr_in)
                );
            } else {
                ret = affine_apply_homo_flint(
                    (flint*) PyArray_DATA(arr_out),
                    at_self->array,
                    (flint*) PyArray_DATA(arr_in),
                    PyArray_NDIM(arr_in),
                    PyArray_SHAPE(arr_in),
                    PyArray_STRIDES(arr_in)
                );
            }
        }
        else {
            arr_in = (PyArrayObject*) PyArray_FROM_OT(arg, NPY_FLINT);
            if (arr_in == NULL) {
                PyErr_SetString(PyExc_TypeError, "Could not read input as 3x? or 4x? array of numbers");
                Py_DECREF(arg);
                Py_DECREF(arr_out);
                return NULL;
            }
            if (arr_out_shape[0] == 3) {
                ret = affine_apply_vert_flint(
                    (flint*) PyArray_DATA(arr_out),
                    at_self->array,
                    (flint*) PyArray_DATA(arr_in),
                    PyArray_NDIM(arr_in),
                    PyArray_SHAPE(arr_in),
                    PyArray_STRIDES(arr_in)
                );
            } else {
                ret = affine_apply_homo_flint(
                    (flint*) PyArray_DATA(arr_out),
                    at_self->array,
                    (flint*) PyArray_DATA(arr_in),
                    PyArray_NDIM(arr_in),
                    PyArray_SHAPE(arr_in),
                    PyArray_STRIDES(arr_in)
                );
            }
        }
        if (ret < 0) {
            PyErr_SetString(PyExc_ValueError, "Array has too many (>10) dimensions");
            Py_DECREF(arr_in);
            Py_DECREF(arr_out);
            return NULL;
        }
        Py_DECREF(arr_in);
    } else {
        arr_in = (PyArrayObject*) PyArray_FROM_OT(arg, NPY_FLINT);
        if (arr_in == NULL) {
            PyErr_SetString(PyExc_TypeError, "Could not read input as 3x? or 4x? array of numbers");
            Py_DECREF(arg);
            Py_DECREF(arr_out);
            return NULL;
        }
        Py_INCREF(arr_in);
        nd = PyArray_NDIM(arr_in);
        if (nd < 1) {
            PyErr_SetString(PyExc_ValueError, "Argument should be an affine transform or 3x? or 4x? array");
            Py_DECREF(arr_in);
            return NULL;
        }
        arr_out_shape = PyArray_SHAPE(arr_in);
        if (arr_out_shape[0] != 3 || arr_out_shape[0] != 4) {
            PyErr_SetString(PyExc_ValueError, "Argument should be an affine transform or 3x? or 4x? array");
            Py_DECREF(arr_in);
            return NULL;
        }
        arr_out = (PyArrayObject*) PyArray_SimpleNew(nd, arr_out_shape, NPY_FLINT);
        if (arr_out == NULL) {
            PyErr_SetString(PyExc_SystemError, "Error allocating output array");
            Py_DECREF(arr_in);
            return NULL;
        }
        if (arr_out_shape[0] == 3) {
            ret = affine_apply_vert_flint(
                (flint*) PyArray_DATA(arr_out),
                at_self->array,
                (flint*) PyArray_DATA(arr_in),
                PyArray_NDIM(arr_in),
                PyArray_SHAPE(arr_in),
                PyArray_STRIDES(arr_in)
            );
        } else {
            ret = affine_apply_homo_flint(
                (flint*) PyArray_DATA(arr_out),
                at_self->array,
                (flint*) PyArray_DATA(arr_in),
                PyArray_NDIM(arr_in),
                PyArray_SHAPE(arr_in),
                PyArray_STRIDES(arr_in)
            );
        }
        if (ret < 0) {
            PyErr_SetString(PyExc_ValueError, "Array has too many (>10) dimensions");
            Py_DECREF(arr_in);
            Py_DECREF(arr_out);
            return NULL;
        }
        Py_DECREF(arr_in);
    }
    return (PyObject*) arr_out;
}

/// @brief Defines the methods for Affine Transforms
static PyMethodDef pyaffine_methods[] = {
    // Pickle support functions
    {"from_mat", pyaffine_from_mat, METH_CLASS | METH_VARARGS,
    from_mat_docstring},
    {"Translation", pyaffine_translation, METH_CLASS | METH_VARARGS | METH_KEYWORDS,
    translation_docstring},
    {"Scale", pyaffine_scale, METH_CLASS | METH_VARARGS| METH_KEYWORDS,
    scale_docstring},
    {"Rotation", pyaffine_rotation, METH_CLASS | METH_VARARGS| METH_KEYWORDS,
    rotation_docstring},
    {"Reflection", pyaffine_reflection, METH_CLASS | METH_VARARGS| METH_KEYWORDS,
    reflection_docstring},
    {"Skew", pyaffine_skew, METH_CLASS | METH_VARARGS| METH_KEYWORDS,
    skew_docstring},
    {"apply", pyaffine_apply, METH_O,
    apply_docstring},
    // sentinel
    {NULL, NULL, 0, NULL}
};

/// @brief The Custom type structure for the new AffineTransform object
static PyTypeObject PyAffine_Type = {
    PyVarObject_HEAD_INIT(NULL, 0) // PyObject_VAR_HEAD
    .tp_name = "AffineTransform", // const char *tp_name; /* For printing, in format "<module>.<name>" */
    .tp_doc = "4x4 Affine transform matrix of flints",// const char *tp_doc; /* Documentation string */
    .tp_basicsize = sizeof(PyAffine), //Py_ssize_t tp_basicsize, tp_itemsize; /* For allocation */
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // unsigned long tp_flags; /* Flags to define presence of optional/expanded features */
    // struct _typeobject *tp_base; Used if inheriting from other class
    .tp_new = pyaffine_new, //newfunc tp_new;
    .tp_init = pyaffine_init, // initproc tp_init;
    .tp_repr = pyaffine_str, // reprfunc tp_repr;
    .tp_str = pyaffine_str, // reprfunc tp_str;
    .tp_hash = pyaffine_hash, // hashfunc tp_hash;
    .tp_getset = pyaffine_getset, // struct PyGetSetDef *tp_getset;
    .tp_methods = pyaffine_methods, // struct PyMethodDef *tp_methods;
    // unsigned int tp_version_tag;
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "affine",
    .m_doc = "Affine Transforms",
    .m_size = -1
};

/// @brief The module initialization function
PyMODINIT_FUNC PyInit_affine(void) {
    PyObject* m;
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "Could not create affine module.");
        return NULL;
    }
    // Import and initialize nmpy
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
    // Register the new AffineTransform type
    if (PyType_Ready(&PyAffine_Type) < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "Could not initialize flint type.");
        return NULL;
    }
    Py_INCREF(&PyAffine_Type);
    if (PyModule_AddObject(m, "AffineTransform", (PyObject *) &PyAffine_Type) < 0) {
        Py_DECREF(&PyAffine_Type);
        Py_DECREF(m);
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "Could not add affine.AffineTransform type to module.");
        return NULL;
    }
    return m;
}