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
    /// CsgNode children
    vec children;
    /// affine transformations
    vec affine_transform;
    // CsgSolid objects
    vec csg_solid;
} CsgNode;

int csg_new(CsgNode* csg_node, CsgNodeType csg_type) {
    csg_node->t = csg_type;
    vec_new(&children, sizeof(CsgNode*));
    vec_new(&affine_transform, sizeof(AffineTrans));
    vec_new(&csg_solid, sizeof(CsgSolid));
}

#ifdef __cplusplus
}
#endif

#endif //PYSCAD_UTILS_H
