/// @file affine.c 
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

#include <flint.h>

#include "affine.h"

/// @brief Set the 3x3 components of the array as an x axis rotation matrix
void affine_set_rotaa(flint* self, flint* axis, flint angle) {
    affine_eye(self);
    int i;
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
    self[0] = flint_add(c, flint_multiply(flint_multiply(u[0], u[0]), omc));
    self[5] = flint_add(c, flint_multiply(flint_multiply(u[1], u[1]), omc));
    self[10] = flint_add(c, flint_multiply(flint_multiply(u[2], u[2]), omc));
    // off-diagonal-z special
    a = flint_multiply(flint_multiply(u[1], u[0]), omc);
    b = flint_multiply(u[2], s);
    self[1] = flint_subtract(a, b);
    self[4] = flint_add(a, b);
    // off-diagonal-y special
    a = flint_multiply(flint_multiply(u[0], u[2]), omc);
    b = flint_multiply(u[1], s);
    self[2] = flint_add(a, b);
    self[8] = flint_subtract(a, b);
    // off-diagonal-x special
    a = flint_multiply(flint_multiply(u[1], u[2]), omc);
    b = flint_multiply(u[0], s);
    self[6] = flint_subtract(a, b);
    self[9] = flint_add(a, b);
}

/// @brief Reflection through arbitrary plane specified by unit vector through the origin
void affine_set_refl_u(flint* self, flint* unitvec) {
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
    affine_eye(self);
    for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
            a = flint_multiply(int_to_flint(2), flint_multiply(u[i], u[j]));
            flint_inplace_subtract(&(self[4*i+j]), a);
        }
    }
}

/// @brief Create a skew matrix with skew plane normal n, and 
void affine_set_skew(flint* self, flint* n, flint* s) {
    int i, j;
    flint _n[3], _s[3];
    // Normalize n, and get s_perp (s-nhat x nhat^T.)
    flint sum = int_to_flint(0);
    flint dot = int_to_flint(0);
    for (i=0; i<3; i++) {
        _n[i] = n[i];
        _s[i] = s[i];
        flint_inplace_add(&sum, flint_multiply(_n[i], _n[i]));
        flint_inplace_add(&dot, flint_multiply(_n[i], _s[i]));
    }
    if (!flint_eq(sum, int_to_flint(1))) {
        sum = flint_sqrt(sum);
        flint_inplace_divide(&dot, sum);
        for (i=0; i<3; i++) {
            flint_inplace_divide(&(_n[i]), sum);
        }
    }
    for (i=0; i<3; i++) {
        flint_inplace_subtract(&(_s[i]), flint_multiply(dot, _n[i]));
    }
    affine_eye(self);
    for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
            flint_inplace_add(&(self[4*i+j]), flint_multiply(_s[i],_n[j]));
        }
    }
}

/// @brief Combine two affine transforms to make a third
void affine_combine(flint* out, flint* lhs, flint* rhs) {
    int i, j, k;
    for (i=0; i<4; i++) {
        for (j=0; j<4; j++) {
            out[4*i+j] = int_to_flint(0);
            for (k=0; k<4; k++) {
                flint_inplace_add(&(out[4*i+j]), flint_multiply(lhs[4*i+k], rhs[4*k+j]));
            }
        }
    }
}

/// @brief Maximum number of dimension for an array of vertices
#define MAX_DIMS 10
/// @brief
inline flint flint_identity(flint f){ return f; }
/// @brief
#define AFFINE_APPLY_VERT(TYPE, TYPE_CONV_FUNC) \
int affine_apply_vert_##TYPE(flint* vert_out, flint* affine, TYPE* vert_in,\
                             int ndims, int* dims, int* strides) {\
    int i, j, k;\
    int idx[MAX_DIMS];\
    uint8_t* cur_vertex_ptr;\
    int num_verts;\
    int offset;\
    int out_idx;\
    TYPE vertex_coord;\
    flint vk;\
    /* Check inputs for valid structure */\
    if (ndims == 0 || dims[0] != 3 || ndims > MAX_DIMS) {\
        return -1;\
    }\
    /* Initialize data */\
    num_verts = 1;\
    idx[0] = 0;\
    for (i=1; i<ndims; i++) {\
        idx[i] = 0;\
        num_verts *= dims[i];\
    }\
    /* outer loop over vertices */\
    offset = 0;\
    for (i=0; i<num_verts; i++) {\
        cur_vertex_ptr = (((uint8_t*) vert_in) + offset);\
        /* inner loop over for matrix multiplication */\
        for (j=0; j<3; j++) {\
            out_idx = num_verts*i+dims[0]*j;\
            vert_out[out_idx] = int_to_flint(0);\
            for (k=0; k<3; k++) {\
                vertex_coord = *((TYPE*) (cur_vertex_ptr + k*strides[0]));\
                vk = TYPE_CONV_FUNC(vertex_coord);\
                flint_inplace_add(&(vert_out[out_idx]), flint_multiply(affine[4*j+k], vk));\
            }\
            flint_inplace_add(&(vert_out[out_idx]), affine[4*j+3]);\
        }\
        /* increment the idx array and offset */\
        for (j=ndims-1; j>0; j--) {\
            idx[j]++;\
            offset += strides[j];\
            if (idx[j] < dims[j]) {\
                break;\
            }\
            idx[j] = 0;\
            offset -= dims[j]*strides[j];\
        }\
    }\
    return 0;\
}

#define AFFINE_APPLY_HOMO(TYPE, TYPE_CONV_FUNC)\
int affine_array_homo_##TYPE(flint* homo_out, flint* affine, TYPE* homo_in,\
                             int ndims, int* dims, int* strides) {\
    int i, j, k;\
    int idx[MAX_DIMS];\
    uint8_t* cur_vertex_ptr;\
    int num_verts;\
    int offset;\
    int out_idx;\
    TYPE vertex_coord;\
    flint vk;\
    /* Check inputs for valid structure */\
    if (ndims == 0 || dims[0] != 4 || ndims > MAX_DIMS) {\
        return -1;\
    }\
    /* Initialize data */\
    num_verts = 1;\
    idx[0] = 0;\
    for (i=1; i<ndims; i++) {\
        idx[i] = 0;\
        num_verts *= dims[i];\
    }\
    /* outer loop over vertices */\
    offset = 0;\
    for (i=0; i<num_verts; i++) {\
        cur_vertex_ptr = (((uint8_t*) homo_in) + offset);\
        /* inner loop over for matrix multiplication */\
        for (j=0; j<4; j++) {\
            out_idx = num_verts*i+dims[0]*j;\
            homo_out[out_idx] = int_to_flint(0);\
            for (k=0; k<4; k++) {\
                vertex_coord = *((TYPE*) (cur_vertex_ptr + k*strides[0]));\
                vk = TYPE_CONV_FUNC(vertex_coord);\
                flint_inplace_add(&(homo_out[out_idx]), flint_multiply(affine[4*j+k], vk));\
            }\
        }\
        out_idx = num_verts*i+dims[0]*3;\
        if ( !flint_eq(homo_out[out_idx], int_to_flint(1))) {\
            vk = homo_out[out_idx];\
            for (j=0; j<3; j++) {\
                out_idx = num_verts*i+dims[0]*j;\
                flint_inplace_divide(&(homo_out[out_idx]), vk);\
            }\
        }\
        /* increment the idx array and offset */\
        for (j=ndims-1; j>0; j--) {\
            idx[j]++;\
            offset += strides[j];\
            if (idx[j] < dims[j]) {\
                break;\
            }\
            idx[j] = 0;\
            offset -= dims[j]*strides[j];\
        }\
    }\
    return 0;\
}

AFFINE_APPLY_VERT(int, int_to_flint)
AFFINE_APPLY_VERT(double, double_to_flint)
AFFINE_APPLY_VERT(flint, flint_identity)

AFFINE_APPLY_HOMO(int, int_to_flint)
AFFINE_APPLY_HOMO(double, double_to_flint)
AFFINE_APPLY_HOMO(flint, flint_identity)
