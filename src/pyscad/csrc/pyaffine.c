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

static inline flint flint_identity(flint f){ return f; }

PyDoc_STRVAR(apply_vert_docstring, "\
Apply an affine transform to an array of 3-length vertices");
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
        if (!flint_eq(w, int_to_flint(1))) {
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

static PyMethodDef AffineMethods[] = {
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
    PyObject* d;
    PyObject* rescale_ufunc;
    PyObject* apply_vert_ufunc;
    PyObject* apply_homo_ufunc;
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
    // Register the ufuncs
    rescale_ufunc = PyUFunc_FromFuncAndDataAndSignature(
        NULL, NULL, NULL, 0, 1, 1, PyUFunc_None,
        "rescale", rescale_docstring, 0, "(4)->(4)");
    int pyaffine_rescale_types[] = {NPY_FLINT, NPY_FLINT};
    PyUFunc_RegisterLoopForType(
        (PyUFuncObject*) rescale_ufunc, NPY_FLINT,
        &pyaffine_rescale_homo, pyaffine_rescale_types, NULL);
    d = PyModule_GetDict(m);
    PyDict_SetItemString(d, "rescale", rescale_ufunc);
    Py_DECREF(rescale_ufunc);
    // Register the ufuncs
    apply_vert_ufunc = PyUFunc_FromFuncAndDataAndSignature(
        NULL, NULL, NULL, 0, 2, 1, PyUFunc_None,
        "apply_vert", apply_vert_docstring, 0, "(4,4),(3)->(3)");
    int pyaffine_apply_vert_types[] = {NPY_FLINT, NPY_FLINT, NPY_FLINT};
    PyUFunc_RegisterLoopForType(
        (PyUFuncObject*) apply_vert_ufunc, NPY_FLINT,
        &pyaffine_apply_vert, pyaffine_apply_vert_types, NULL);
    d = PyModule_GetDict(m);
    PyDict_SetItemString(d, "apply_vert", apply_vert_ufunc);
    Py_DECREF(apply_vert_ufunc);

    return m;
}