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

#ifndef PYSCAD_AFFINE_H
#define PYSCAD_AFFINE_H

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Set the array members to the identity
static inline void affine_eye(flint* self) {
    // Creat the identity transform
    int i;
    for (i=0; i<16; i++) {
        self[i] = int_to_flint(0);
    }
    for (i=0 ; i<16; i+=5) {
        self[i] = int_to_flint(1);
    }
}

/// @brief Set the translation components to the array
static inline void affine_set_translation(flint* self, flint* arr) {
    int i;
    affine_eye(self);
    for (i=0; i<3; i++) {
        self[4*i+3] = arr[i];
    }
}

/// @brief Create a scaling matrix from [sx, sy, sz] values
static inline void affine_set_scale(flint* self, flint* scale) {
    int i;
    affine_eye(self);
    for (i=0; i<3; i++) {
        self[4*i+i] = scale[i];
    }
}

/// @brief Set the 3x3 components of the array as an x axis rotation matrix
static inline void affine_set_rotx(flint* self, flint angle) {
    affine_eye(self);
    flint zero = int_to_flint(0);
    flint one = int_to_flint(1);
    flint c = flint_cos(angle);
    flint s = flint_sin(angle);
    flint ns = flint_negative(s);
    self[0] = one; self[1] = zero; self[2] = zero;
    self[4] = zero; self[5] = c; self[6] = ns;
    self[8] = zero; self[9] = s; self[10] = c;
}

/// @brief Set the 3x3 components of the array as an y axis rotation matrix
static inline void affine_set_roty(flint* self, flint angle) {
    affine_eye(self);
    flint zero = int_to_flint(0);
    flint one = int_to_flint(1);
    flint c = flint_cos(angle);
    flint s = flint_sin(angle);
    flint ns = flint_negative(s);
    self[0] = c; self[1] = zero; self[2] = s;
    self[4] = zero; self[5] = one; self[6] = zero;
    self[8] = ns; self[9] = zero; self[10] = c;
}

/// @brief Set the 3x3 components of the array as an z axis rotation matrix
static inline void affine_set_rotz(flint* self, flint angle) {
    affine_eye(self);
    flint zero = int_to_flint(0);
    flint one = int_to_flint(1);
    flint c = flint_cos(angle);
    flint s = flint_sin(angle);
    flint ns = flint_negative(s);
    self[0] = c; self[1] = ns; self[2] = zero;
    self[4] = s; self[5] = c; self[6] = zero;
    self[8] = zero; self[9] = zero; self[10] = one;
}

/// @brief Set the 3x3 components of the array as an x axis rotation matrix
void affine_set_rotaa(flint* self, flint* axis, flint angle);

/// @brief Reflection through y-z plane
static inline void affine_set_refl_yz(flint* self) {
    affine_eye(self);
    self[0] = int_to_flint(-1);
}

/// @brief Reflection through z-x plane
static inline void affine_set_refl_zx(flint* self) {
    affine_eye(self);
    self[5] = int_to_flint(-1);
}

/// @brief Reflection through x-y plane
static inline void affine_set_refl_xy(flint* self) {
    affine_eye(self);
    self[10] = int_to_flint(-1);
}

/// @brief Reflection through arbitrary plane specified by unit vector through the origin
void affine_set_refl_u(flint* self, flint* unitvec);

/// @brief Create a skew matrix with skew plane normal n, and 
void affine_set_skew(flint* self, flint* n, flint* s);

/// @brief Relocate the center of a linear transformation
static inline void affine_relocate_center(flint* self, flint* c) {
    int i, j;
    flint b[3];
    for (i=0; i<3; i++) {
        b[i] = int_to_flint(0);
        for (j=0; j<3; j++) {
            flint_inplace_add(&(b[i]), flint_multiply(self[4*i+j],c[j]));
        }
        self[4*i+3] = flint_subtract(c[i], b[i]);
    }
}

#ifdef __cplusplus
}
#endif

#endif // PYSCAD_AFFINE_H