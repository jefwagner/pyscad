/// @file curve.c A space-curve object
//
// Copyright (c) 2023, Jef Wagner <jefwagner@gmail.com>
//
// This file is part of pyscad.
//
// Pyscad is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// Pyscad is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// Pyscad. If not, see <https://www.gnu.org/licenses/>.
//

#include <Python.h>

typedef struct {
    PyObject_HEAD
} PyCurve;

static PyTypeObject PyCurve_Type;

