/// @file affine.h 
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

/// @brief The __new__ allocating constructor
/// @param type The type of the PyObject
/// @return A new PyObject of type `type`
static PyObject* pyaffine_new(PyTypeObject* type, 
                              PyObject* NPY_UNUSED(args),
                              PyObject* NPY_UNUSED(kwargs)) {
    PyAffine* self = (PyAffine*) type->tp_alloc(type, 0);
    return (PyObject*) self;
}

/// @brief Set the array members to the identity
static void pyaffine_eye(PyAffine* self) {
    // Creat the identity transform
    int i;
    for (i=0; i<16; i++) {
        self->array[i] = int_to_flint(0);
    }
    for (i=0 ; i<16; i+=5) {
        self->array[i] = int_to_flint(1);
    }
}

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

/// @brief Set the translation components to the array
static void pyaffine_set_translation(PyAffine* self, PyArrayObject* arr) {
    pyaffine_eye(self);
    self->array[3] = *((flint*) PyArray_GETPTR1(arr, 0));
    self->array[7] = *((flint*) PyArray_GETPTR1(arr, 1));
    self->array[11] = *((flint*) PyArray_GETPTR1(arr, 2));
}

/// @brief Create a scaling matrix from [sx, sy, sz] values
static void pyaffine_set_scale(PyAffine* self, flint* scale) {
    int i;
    pyaffine_eye(self);
    for (i=0; i<3; i++) {
        self->array[4*i+i] = scale[i];
    }
}

/// @brief Create a scaling matrix from a [sx, sy, sz] centered on [cx, cy, cz]
static void pyaffine_set_scale_p(PyAffine* self, flint* scale, flint* center) {
    int i;
    pyaffine_eye(self);
    for (i=0; i<3; i++) {
        self->array[4*i+i] = scale[i];
        self->array[4*i+3] = flint_subtract(center[i], flint_multiply(scale[i], center[i]));
    }
}

/// @brief Set the 3x3 components of the array as an x axis rotation matrix
static void pyaffine_set_rotx(PyAffine* self, flint angle) {
    pyaffine_eye(self);
    flint* arr = self->array;
    flint zero = int_to_flint(0);
    flint one = int_to_flint(1);
    flint c = flint_cos(angle);
    flint s = flint_sin(angle);
    flint ns = flint_negative(s);
    arr[0] = one; arr[1] = zero; arr[2] = zero;
    arr[4] = zero; arr[5] = c; arr[6] = ns;
    arr[8] = zero; arr[9] = s; arr[10] = c;
}

/// @brief Set the 3x3 components of the array as an y axis rotation matrix
static void pyaffine_set_roty(PyAffine* self, flint angle) {
    pyaffine_eye(self);
    flint* arr = self->array;
    flint zero = int_to_flint(0);
    flint one = int_to_flint(1);
    flint c = flint_cos(angle);
    flint s = flint_sin(angle);
    flint ns = flint_negative(s);
    arr[0] = c; arr[1] = zero; arr[2] = s;
    arr[4] = zero; arr[5] = one; arr[6] = zero;
    arr[8] = ns; arr[9] = zero; arr[10] = c;
}

/// @brief Set the 3x3 components of the array as an z axis rotation matrix
static void pyaffine_set_rotz(PyAffine* self, flint angle) {
    pyaffine_eye(self);
    flint* arr = self->array;
    flint zero = int_to_flint(0);
    flint one = int_to_flint(1);
    flint c = flint_cos(angle);
    flint s = flint_sin(angle);
    flint ns = flint_negative(s);
    arr[0] = c; arr[1] = ns; arr[2] = zero;
    arr[4] = s; arr[5] = c; arr[6] = zero;
    arr[8] = zero; arr[9] = zero; arr[10] = one;
}

/// @brief Set the 3x3 components of the array as an x axis rotation matrix
static void pyaffine_set_rotaa(PyAffine* self, flint* axis, flint angle) {
    pyaffine_eye(self);
    int i;
    flint* arr = self->array;
    flint one = int_to_flint(1);
    flint c = flint_cos(angle);
    flint omc = flint_subtract(one, c);
    flint s = flint_sin(angle);
    flint u[3];
    flint sum = int_to_flint(0);
    // Get unit vector and square length
    for (i=0; i<3; i++) {
        u[i] = axis[i];
        flint_inplace_add(&sum, flint_multiply(u[i], u[i]));
    }
    // normalize if required
    if (!flint_eq(sum, int_to_flint(1))) {
        sum = flint_sqrt(sum);
        for (i=0; i<3; i++) {
            flint_inplace_divide(&(u[i]), sum);
        }
    }
    flint a, b;
    // diagnonal
    arr[0] = flint_add(c, flint_multiply(flint_multiply(u[0], u[0]), omc));
    arr[5] = flint_add(c, flint_multiply(flint_multiply(u[1], u[1]), omc));
    arr[10] = flint_add(c, flint_multiply(flint_multiply(u[2], u[2]), omc));
    // off-diagonal-z special
    a = flint_multiply(flint_multiply(u[1], u[0]), omc);
    b = flint_multiply(u[2], s);
    arr[1] = flint_subtract(a, b);
    arr[4] = flint_add(a, b);
    // off-diagonal-y special
    a = flint_multiply(flint_multiply(u[0], u[2]), omc);
    b = flint_multiply(u[1], s);
    arr[2] = flint_add(a, b);
    arr[8] = flint_subtract(a, b);
    // off-diagonal-x special
    a = flint_multiply(flint_multiply(u[1], u[2]), omc);
    b = flint_multiply(u[0], s);
    arr[6] = flint_subtract(a, b);
    arr[9] = flint_add(a, b);
}

/// @brief Reflection through y-z plane
static void pyaffine_set_refl_yz(PyAffine* self) {
    pyaffine_eye(self);
    self->array[0] = int_to_flint(-1);
}

/// @brief Reflection through z-x plane
static void pyaffine_set_refl_zx(PyAffine* self) {
    pyaffine_eye(self);
    self->array[5] = int_to_flint(-1);
}

/// @brief Reflection through x-y plane
static void pyaffine_set_refl_xy(PyAffine* self) {
    pyaffine_eye(self);
    self->array[10] = int_to_flint(-1);
}

/// @brief Reflection through arbitrary plane specified by unit vector through the origin
static void pyaffine_set_refl_u(PyAffine* self, flint* unitvec) {
    int i, j;
    flint a;
    flint u[3];
    flint sum = int_to_flint(0);
    // Get unit vector and square length
    for (i=0; i<3; i++) {
        u[i] = unitvec[i];
        flint_inplace_add(&sum, flint_multiply(u[i], u[i]));
    }
    // normalize if required
    if (!flint_eq(sum, int_to_flint(1))) {
        sum = flint_sqrt(sum);
        for (i=0; i<3; i++) {
            flint_inplace_divide(&(u[i]), sum);
        }
    }
    // Set 3x3 component of matrix to Householder transformation I-2v.vT
    pyaffine_eye(self);
    for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
            a = flint_multiply(int_to_flint(2), flint_multiply(u[i], u[j]));
            flint_inplace_subtract(&(self->array[4*i+j]), a);
        }
    }
}

/// @brief Create a shearing matrix that leaves x coorindates alone and moves y and z
static void pyaffine_set_shear_x(PyAffine* self, flint sy, flint sz) {
    pyaffine_eye(self);
    self->array[4*1+0] = sy;
    self->array[4*2+0] = sz;
}

/// @brief Create a shearing matrix that leaves y coorindates alone and moves x and z
static void pyaffine_set_shear_y(PyAffine* self, flint sx, flint sz) {
    pyaffine_eye(self);
    self->array[4*0+1] = sx;
    self->array[4*2+1] = sz;
}

/// @brief Create a shearing matrix that leaves z coorindates alone and moves x and y
static void pyaffine_set_shear_z(PyAffine* self, flint sx, flint sy) {
    pyaffine_eye(self);
    self->array[4*0+2] = sx;
    self->array[4*1+2] = sy;
}

/// @brief Relocate the center of a linear transformation
static void pyaffine_relocate_center(PyAffine* self, flint* c) {
    int i, j;
    flint b[3];
    flint* arr = self->array;
    for (i=0; i<3; i++) {
        b[i] = int_to_flint(0);
        for (j=0; j<3; j++) {
            flint_inplace_add(&(b[i]), flint_multiply(arr[4*i+j],c[j]));
        }
        arr[4*i+3] = flint_subtract(c[i], b[i]);
    }
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
    pyaffine_eye(at_self);
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
    PyAffine* at_ref = NULL;
    PyArrayObject* d_arr = NULL;
    PyObject* d_pyo = NULL;
    PyObject* c_pyo = NULL;

    if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O", translation_keywords, &d_pyo, &c_pyo)) {
        Py_INCREF(d_pyo);
        at_ref = (PyAffine*) pyaffine_new((PyTypeObject*) cls, NULL, NULL);
        if (at_ref == NULL) {
            PyErr_SetString(PyExc_SystemError, "Error allocating new AffineTransform");
            return NULL;
        }
        d_arr = (PyArrayObject*) PyArray_FROM_OT(d_pyo, NPY_FLINT);
        if (d_arr != NULL) {
            if (PyArray_NDIM(d_arr) == 1) {
                if (PyArray_SHAPE(d_arr)[0] == 3) {
                    at_ref->type = AT_TRANSLATION;
                    pyaffine_eye(at_ref);
                    pyaffine_set_translation(at_ref, d_arr);
                    Py_DECREF(d_pyo);
                    Py_DECREF(d_arr);
                    return (PyObject*) at_ref;
                }
            }
        }
    }
    PyErr_SetString(PyExc_ValueError, "Translation argument should be a 3 length sequence");
    Py_XDECREF(d_pyo);
    Py_XDECREF(d_arr);
    return NULL;
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
        // Set the affine transform matrix values
        if (valid_scale) {
            if (use_center) {
                pyaffine_set_scale_p(at_ref, s, c);
            } else {
                pyaffine_set_scale(at_ref, s);
            }
            at_ref->type = AT_SCALE;
            return (PyObject*) at_ref;
        }
    }
    PyErr_SetString(PyExc_ValueError, "s must be a scalar or scalar or 3-length non-uniform scaling [sx, sy, sz]");
    Py_DECREF(at_ref);
    return NULL;
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
            pyaffine_set_rotaa(at_ref, a, th);
            break;
        }
        case 'x': {
            pyaffine_set_rotx(at_ref, th);
            break;
        }
        case 'y': {
            pyaffine_set_roty(at_ref, th);
            break;
        }
        case 'z': {
            pyaffine_set_rotz(at_ref, th);
            break;
        }
    }
    if (use_center) {
        pyaffine_relocate_center(at_ref, c);
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
            pyaffine_set_refl_u(at_ref, n);
            break;
        }
        case 'x': {
            pyaffine_set_refl_yz(at_ref);
            break;
        }
        case 'y': {
            pyaffine_set_refl_zx(at_ref);
            break;
        }
        case 'z': {
            pyaffine_set_refl_xy(at_ref);
            break;
        }
    }
    if (use_center) {
        pyaffine_relocate_center(at_ref, c);
    }
    return (PyObject*) at_ref;
}

/// @brief Create a new pure axis aligned skew AffineTransform
/// @param args[0] axis to leave unchanged 'x', 'y', 'z'
/// @param args[1] skew of 'first' coord
/// @param args[2] skew of 'second' coord
static const char skew_docstring[] = "\
Create a new pure skew transformation.\n\
\n\
:param uaxis: The character 'x','y','z' or a 3 length [ux, uy, uz] vector for\n\
the unaffected axis.\n\
:param sdir: The character 'x','y','z' or a 3 length [sx, sy, sz] vector for\n\
the skew direction.\n\
:param center: Optional 3-length center position [cx, cy, cz] for the center of\n\
the skew operation.\n\
\n\
:return: A skew AffineTransformation.";
static PyObject* pyaffine_skew(PyObject* cls, PyObject* args) {
    PyErr_SetString(PyExc_ValueError, "Not implemented");
    return NULL;
}


/// @brief Combine two affine transforms through matrix multiplication
/// @param O_lhs The left hand side affine transform object
/// @param O_rhs The right hand side affine transform object
/// @return A new affine transform object combining the two inputs
static PyObject* pyaffine_apply_transform(PyObject* O_lhs, PyObject* O_rhs) {
    int i, j, k;
    flint *res;
    flint* lhs = ((PyAffine*) O_lhs)->array;
    flint* rhs = ((PyAffine*) O_rhs)->array;
    PyAffine* result = (PyAffine*) pyaffine_new(&PyAffine_Type, NULL, NULL);
    if (result == NULL) {
        PyErr_SetString(PyExc_SystemError, "Error allocating result AffineTransform");
        return NULL;
    }
    result->type = AT_GENERIC;
    res = result->array;
    for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
            res[4*i+j] = int_to_flint(0);
            for (k=0; k<4; k++) {
                flint_inplace_add(&(res[4*i+j]), flint_multiply(lhs[4*i+k], rhs[4*k+j]));
            }
        }
    }
    return (PyObject*) result;
}

/// @brief Apply the affine transformation to a 3xN? numpy array of flints
/// This 
static PyObject* pyaffine_apply_vertices(PyObject* trans, PyArrayObject* in_arr) {
    /// Affine transform 4x4 matrix
    flint* trans_data = ((PyAffine*) trans)->array;
    /// Input array variables
    int ndim = PyArray_NDIM(in_arr);
    npy_intp* shape = PyArray_SHAPE(in_arr);
    /// Outer loop variables
    int i;
    int num_verts = 1;
    npy_intp outer_stride = PyArray_STRIDE(in_arr, 0)/sizeof(flint);
    // Inner loop variables
    int j, k;
    npy_intp inner_stride = PyArray_STRIDE(in_arr, ndim-1)/sizeof(flint);
    flint homo[4];
    flint one = int_to_flint(1);
    // working data arrays and new object
    flint* in_data = (flint*) PyArray_DATA(in_arr);
    flint* res_data;
    PyArrayObject* result = (PyArrayObject*) PyArray_NewLikeArray(in_arr, NPY_KEEPORDER, NULL, 0);
    if (result == NULL) {
        PyErr_SetString(PyExc_SystemError, "Could not create output array");
        return NULL;
    }
    res_data = (flint*) PyArray_DATA(result);
    // Get the number of vertices we're operating on
    for (i=1; i<ndim; i++) {
        num_verts *= shape[i];
    }
    // Loop over the vertices
    for (i=1; i<num_verts; i++) {
        // Do the matrix-vector multiplication
        for (j=0; j<4; j++) {
            homo[j] = int_to_flint(0);
            for (k=0; k<3; k++) {
                flint_inplace_add(&(homo[j]), flint_multiply(trans_data[4*j+k], in_data[k*outer_stride+i*inner_stride]));
            }
            flint_inplace_add(&(homo[j]), trans_data[4*j+3]);
        }
        // Possibly rescale homogenous coordinates
        if (!flint_eq(homo[3], one)) {
            for (j=0; j<3; j++) {
                flint_inplace_divide(&(homo[j]), homo[3]);
            }
        }
        // Copy data to output
        for (j=0; j<3; j++) {
            res_data[j*outer_stride+i*inner_stride] = homo[j];
        }        
    }

    return (PyObject*) result;
}

/// @brief Apply the affine transformation to a 4xN? numpy array of flints
static PyObject* pyaffine_apply_homogenous(PyObject* trans, PyArrayObject* in_arr) {
    /// Affine transform 4x4 matrix
    flint* trans_data = ((PyAffine*) trans)->array;
    /// Input array variables
    int ndim = PyArray_NDIM(in_arr);
    npy_intp* shape = PyArray_SHAPE(in_arr);
    /// Outer loop variables
    int i;
    int num_verts = 1;
    npy_intp outer_stride = PyArray_STRIDE(in_arr, 0)/sizeof(flint);
    // Inner loop variables
    int j, k;
    npy_intp inner_stride = PyArray_STRIDE(in_arr, ndim-1)/sizeof(flint);
    flint one = int_to_flint(1);
    // working data arrays and new object
    flint* homo;
    flint* in_data = (flint*) PyArray_DATA(in_arr);
    flint* res_data;
    PyArrayObject* result = (PyArrayObject*) PyArray_NewLikeArray(in_arr, NPY_KEEPORDER, NULL, 0);
    if (result == NULL) {
        PyErr_SetString(PyExc_SystemError, "Could not create output array");
        return NULL;
    }
    res_data = (flint*) PyArray_DATA(result);
    // Get the number of vertices we're operating on
    for (i=1; i<ndim; i++) {
        num_verts *= shape[i];
    }
    // Loop over the vertices
    for (i=1; i<num_verts; i++) {
        // Do the matrix-vector multiplication
        for (j=0; j<4; j++) {
            homo[j] = int_to_flint(0);
            for (k=0; k<4; k++) {
                flint_inplace_add(&(homo[j]), flint_multiply(trans_data[4*j+k], in_data[k*outer_stride+i*inner_stride]));
            }
        }
        // Possibly rescale homogenous coordinates
        if (!flint_eq(homo[3], one)) {
            for (j=0; j<3; j++) {
                flint_inplace_divide(&(homo[j]), homo[3]);
            }
            homo[3] = one;
        }
        // Copy data to output
        for (j=0; j<4; j++) {
            res_data[j*outer_stride+i*inner_stride] = homo[j];
        }        
    }

    return (PyObject*) result;
}


/// @brief Apply the affine transformation to an applicable object
/// Apply the affine transform to
/// 1. Another affine transform
/// 2. A 3 length sequence 'vertex' (3) -> (3)
/// 3. A 4 length sequence 'homogenous coordinate vertex' (4) -> (4)
static PyObject* pyaffine_apply(PyObject* self, PyObject* args) {
    PyObject* O = NULL;
    PyArrayObject* arr = NULL;
    PyObject* result = NULL;
    if (PyArg_ParseTuple(args, "O", &O)) {
        Py_XINCREF(O);
        // If the objects is a transform, combine transforms through matrix
        // multiplication
        if (PyObject_IsInstance(O, (PyObject*) &PyAffine_Type)) {
            result = pyaffine_apply_transform(self, O);
        }
        // Otherwise create numpy array of flints from argument
        else {
            arr = (PyArrayObject*) PyArray_FROM_OT(O, NPY_FLINT);
            if (arr != NULL) {
                PyErr_SetString(PyExc_TypeError, "Argument must be an AffineTransform or a numeric sequence type");
                Py_XDECREF(O);
                return NULL;
            }
            if (PyArray_SHAPE(arr)[0] == 3) {
                result = pyaffine_apply_vertices(self, arr);
            } else if (PyArray_SHAPE(arr)[0] == 4) {
                result = pyaffine_apply_homogenous(self, arr);
            } else {
                PyErr_SetString(PyExc_ValueError, "Transforms can only be applied to 3x? array of vertices or 4x? array of homogenous vertices");
            }
        }
        Py_XDECREF(O);
        return (PyObject*) result;
    }
    PyErr_SetString(PyExc_SystemError, "Error allocating result AffineTransform");
    return NULL;
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
    {"Skew", pyaffine_skew, METH_CLASS | METH_VARARGS,
    skew_docstring},
    {"apply", pyaffine_apply, METH_VARARGS,
    "Apply the transformation to a transformation or vertex"},
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