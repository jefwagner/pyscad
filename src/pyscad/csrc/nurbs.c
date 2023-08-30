/// @file nurbs.c 
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

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include <flint.h>
#include <numpy_flint.h>


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_c_nurbs",
    .m_doc = "Nurbs NumPy UFuncs",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__c_nurbs(void) {
    PyObject*m;
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

    return m;
}