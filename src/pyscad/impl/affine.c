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

#include <stdint.h>
#include <string.h>

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
    PyAffineTrans* self = (PyAffineTrans*) type->tp_alloc(type, 0);
    return (PyObject*) self;
}

/// @brief The __init__ initializing constructor
/// @param self The object to be initialized
/// @param args Unused positional argument tuple
/// @param kwargs Unused keyword argument dict
/// @return 0 on success, -1 on failure
static int pyaffine_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyAffineTrans* at_self = (PyAffineTrans*) self;
    int i;

    if ((PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs))){
        PyErr_SetString(PyExc_TypeError,
                        "AffineTrans constructor doesn't take any arguments")
        return -1;
    }

    // Creat the identity transform
    for (i=0; i<16; i++) {
        at_self->array[i] = int_to_flint(0);
    }
    for (i=0 ; i<16; i+=5) {
        at_self->array[i] = int_to_flint(1);
    }

    return 0;
}

/// @brief The __str__ printing method
/// @return A python string representation of the tracked value
static PyObject* pyaffine_str(PyObject* self) {
    PyAffineTrans* at_self = (PyAffineTrans*) self;
    switch(at_self->type) {
        case AT_TRANSLATION:
            return PyUnicode_FromString("Translation");
        case AT_ROTATION:
            return PyUnicode_FromString("Rotation");
        case AT_SCALE:
            return PyUnicode_FromString("Scale");
        case AT_REFLECT:
            return PyUnicode_FromString("Reflection");
        case AT_SKEW:
            return PyUnicode_FromString("SkewTransform");
        default:
            return PyUnicode_FromString("AffineTransform");
    }
}

/// @brief The __hash__ function create an unique-ish integer from the flint.
///        Implements Bob Jenkin's one-at-a-time hash.
/// @return An integer to be used in a hash-table
static Py_hash_t pyaffine_hash(PyObject *self) {
    PyAffineTrans* at_self = (PyAffineTrans*) self;
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
static PyObject* pyaffine_get_array(PyObject *self, void *NPY_UNUSED(closure)) {
    PyAffineTrans* at_self = (PyAffineTrans*) self;
    // Create a new 4x4 numpy array
    PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLINT);
    int nd = 2;
    npy_intp dims[2] = {4,4};
    PyObject* arr = PyArray_NewFromDescr(
        &PyArray_Type, descr, nd, dims, NULL, NULL, NPY_ARRAY_CARRAY, NULL);
    if (arr == NULL) {
        PyErr_SetString(PyExc_SystemError, "Could not get create new numpy array of flints");
        return NULL;
    }
    // Copy data from transform into numpy array
    memcpy(PyArray_DATA(arr), (void*) at_self->array, 16*sizeof(flint));
    return (PyObject*) arr;
}

/// @brief Defines the property getter/setter methods
PyGetSetDef pyaffine_getset[] = {
    {"array", pyaffine_get_array, NULL,
    "Affine transform as 4x4 array of flints", NULL},
    //sentinal
    {NULL, NULL, NULL, NULL, NULL}
};

/// @brief Create a new pure translation AffineTransform
/// @param args The [x,y,z] coordinates of the translation
PyObject* pyaffine_translation(PyObject* cls, PyObject* args) {
    PyAffineTrans* at_ref;
    PyArray_Descr* descr;
    PyArrayObject* arr;
    PyObject O = {0};
    if (PyArg_ParseTuple(args, "O", &O)) {
        at_ref = (PyAffineTrans*) pyaffine_new((PyTypeObject*) cls, NULL, NULL);
        if (at_ref == NULL) {
            PyErr_SetString(PyExc_SystemError, "Error allocating new AffineTransform");
            return NULL;
        }
        if (pyaffine_init(at_ref, NULL, NULL) < 0) {
            PyErr_SetString(PyExc_SystemError, "Error initializing AffineTransform");
            return NULL;
        }
        descr = PyArray_DescrFromType(NPY_FLINT);
        arr = PyArray_FromAny(&O, descr, 1, 1, NPY_ARRAY_CARRAY, NULL);
        if (arr != NULL) {
            if (PyArray_SHAPE(arr)[0] == 3) {
                at_ref->array[3] = *((flint*) PyArray_GETPTR1(arr, 0));
                at_ref->array[7] = *((flint*) PyArray_GETPTR1(arr, 1));
                at_ref->array[11] = *((flint*) PyArray_GETPTR1(arr, 2));
                at_ref->type = AT_TRANSLATION;
                return (PyObject*) at_ref;
            }
            PyErr_SetString(PyExc_ValueError, "Argument must be a 3-length sequence");
            return NULL;
        } 
        PyErr_SetString(PyExc_ValueError, "Argument must be a 3-length sequence");
        return NULL;
    }
}

/// @brief Defines the methods for Affine Transforms
PyMethodDef pyarray_methods[] = {
    // Pickle support functions
    {"Translation", pyarray_translation, METH_CLASS | METH_VARARGS,
    "Create a new translation transform"},
    // sentinel
    {NULL, NULL, 0, NULL}
};

/// @brief The Custom type structure for the new AffineTransform object
static PyTypeObject PyAffine_Type = {
    PyVarObject_HEAD_INIT(NULL, 0) // PyObject_VAR_HEAD
    .tp_name = "AffineTransform", // const char *tp_name; /* For printing, in format "<module>.<name>" */
    .tp_doc = "4x4 Affine transform matrix of flints",// const char *tp_doc; /* Documentation string */
    .tp_basicsize = sizeof(PyAffineTrans), //Py_ssize_t tp_basicsize, tp_itemsize; /* For allocation */
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
PyMODINIT_FUNC PyInit_bar(void) {
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