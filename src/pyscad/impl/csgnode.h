/// @file csgnode.h 
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

#ifndef PYSCAD_UTILS_H
#define PYSCAD_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/// \brief Types of Constructive solid geometry nodes
typedef enum _CsgNodeType{
    None,
    Union,
    Intxn,
    Diff,
    Sphere,
    Box,
    Cyl,
    Extr,
    SolRot,
} CsgNodeType;

/// \brief Data-structure that contains all the data for the Csg Node
typedef struct _CsgNode {
    /// A enum for the type of node
    CsgNodeType t;
    /// max_size, num, and an array of pointers to CsgNode children
    size_t ch_size;
    size_t ch_num;
    struct _CsgNode** children;
    /// max_size, num, and an array of affine transformations
    size_t at_size;
    size_t at_num;
    AffineTrans *affine_trans;
    // max_size, num, and an array of CsgSolid objects
    size_t cs_size;
    size_t cz_num;
    CsgSolid *csg_solid;
} CsgNode;

#ifdef __cplusplus
}
#endif

#endif //PYSCAD_UTILS_H
